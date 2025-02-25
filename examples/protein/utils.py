import logging
import sys
import typing as T
from pathlib import Path
import re
import torch
import math
from collections import Counter
import pandas as pd
from transformers import EsmForProteinFolding
from timeit import default_timer as timer
import numpy as np
from transformers.models.esm.openfold_utils.loss import compute_tm  
# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def calculate_entropy(sequence: str) -> float:
    """Calculate Shannon entropy of a sequence."""
    amino_acid_counts = Counter(sequence)
    total_amino_acids = len(sequence)
    probabilities = (count / total_amino_acids for count in amino_acid_counts.values())
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def read_fasta(path: str, keep_gaps: bool = True, keep_insertions: bool = True, to_upper: bool = False):
    """Read sequences from a FASTA file."""
    with open(path, "r") as f:
        for result in read_alignment_lines(
            f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        ):
            yield result

def read_alignment_lines(lines, keep_gaps=True, keep_insertions=True, to_upper=False):
    """Parse alignment lines from a FASTA file."""
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        if len(line) > 0 and line[0] == ">":
            if seq is not None and 'X' not in seq:
                yield desc, parse(seq)
            desc = line.strip().lstrip(">")
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    if seq is not None and 'X' not in seq:
        yield desc, parse(seq)

def enable_cpu_offloading(model):
    """Enable CPU offloading for the model."""
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))
    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)
    return model

def init_model_on_gpu_with_cpu_offloading(model):
    """Initialize model with CPU offloading."""
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model

def create_batched_sequence_dataset(
    sequences: T.List[T.Tuple[str, str]], 
    max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    """Create batched sequences for efficient processing."""
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)
    if batch_headers:
        yield batch_headers, batch_sequences

def run_esmfold_eval(
    fasta_dir: Path,
    output_dir: Path,
    num_recycles: int = None,
    max_tokens_per_batch: int = 1024,
    chunk_size: int = None,
    cpu_only: bool = False,
    cpu_offload: bool = False,
) -> pd.DataFrame:
    """
    Run ESMFold evaluation on generated sequences.
    
    Args:
        fasta_dir: Directory containing FASTA files
        output_dir: Directory to save PDB files
        num_recycles: Number of recycles for ESMFold
        max_tokens_per_batch: Maximum tokens per batch
        chunk_size: Chunk size for axial attention
        cpu_only: Whether to use CPU only
        cpu_offload: Whether to enable CPU offloading
    
    Returns:
        DataFrame with evaluation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load ESMFold model
    logger.info("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
    model = model.eval()
    # model.set_chunk_size(chunk_size)

    # Set device
    if cpu_only:
        model.esm.float()
        model.cpu()
    elif cpu_offload:
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        model.cuda()

    # Process FASTA files
    data_records = []
    fasta_files = list(fasta_dir.glob("*.fasta"))
    logger.info(f"Found {len(fasta_files)} FASTA files")

    for fasta_path in fasta_files:
        logger.info(f"Processing {fasta_path}")
        pdb_dir = output_dir / fasta_path.stem
        pdb_dir.mkdir(exist_ok=True)

        # Read sequences
        sequences = sorted(read_fasta(str(fasta_path)), key=lambda x: len(x[1]))
        if not sequences:
            continue

        # Process batches
        for headers, seqs in create_batched_sequence_dataset(sequences, max_tokens_per_batch):
            try:
                start_time = timer()
                output = model.infer(seqs,)
                output = {k: v.cpu() for k, v in output.items()}
                ptm = torch.stack(
                    [
                        compute_tm(
                            batch_ptm_logits[None, :sl, :sl],
                            max_bins=31,
                            no_bins=64,
                        )
                        for batch_ptm_logits, sl in zip(output["ptm_logits"], (len(seq) for seq in seqs))
                    ]
                )
                output["ptm"] = ptm
                
                # Generate PDBs and metrics
                pdbs = model.output_to_pdb(output)
                paes = (output["aligned_confidence_probs"].numpy() * 
                       np.arange(64).reshape(1, 1, 1, 64)).mean(-1) * 31
                paes = paes.mean(-1).mean(-1)
                output["mean_plddt"] = 100 * (output["plddt"] * output["atom37_atom_exists"]).sum(dim=(1, 2)) / output["atom37_atom_exists"].sum(dim=(1, 2))
                # Save results
                for header, seq, pdb_str, plddt, ptm, pae in zip(
                    headers, seqs, pdbs, 
                    output["mean_plddt"], output["ptm"], paes
                ):
                    pdb_file = pdb_dir / f"{header}_plddt_{plddt.mean().item():.1f}_ptm_{ptm.item():.3f}_pae_{pae.item():.3f}.pdb"
                    pdb_file.write_text(pdb_str)
                    
                    data_records.append({
                        'FASTA_file': fasta_path.name,
                        'PDB_path': str(pdb_file),
                        'sequence': seq,
                        'Length': len(seq),
                        'pLDDT': plddt.mean().item(),
                        'pTM': ptm.item(),
                        'pAE': pae.item(),
                        'Entropy': calculate_entropy(seq)
                    })

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"CUDA OOM for batch size {len(seqs)}")
                    continue
                raise e

    # Create DataFrame
    df = pd.DataFrame(data_records)
    return df 