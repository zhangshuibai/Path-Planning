import argparse
import os
import time
from pathlib import Path
from pprint import pprint
import math
import torch
from transformers import AutoTokenizer, EsmForMaskedLM, AutoModel
from path_planning.p2 import p2_sampling
from path_planning.scheduler import sine_scheduler
from path_planning.score_function import diff_top2
from path_planning.utils import seed_everything
from utils import run_esmfold_eval

def ignore_special_tokens_logits(logits, tokenizer):
    """
    Masks out the logits of special tokens to prevent them from being sampled.
    
    Args:
        logits (Tensor): Logits output from the model of shape [B, L, V].
        tokenizer: The tokenizer to access special token IDs.
    
    Returns:
        Tensor: Modified logits with special tokens masked out.
    """
    logits[..., tokenizer.mask_token_id] = -math.inf
    logits[..., tokenizer._token_to_id["X"]] = -math.inf
    logits[..., tokenizer.pad_token_id] = -math.inf
    logits[..., tokenizer.cls_token_id] = -math.inf
    logits[..., tokenizer.eos_token_id] = -math.inf
    return logits

class ModelWrapper:
    """Wrapper for the ESM model to handle logits processing."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def __call__(self, x):
        outputs = self.model(x)
        logits = outputs.logits
        return ignore_special_tokens_logits(logits.float(), self.tokenizer)

def create_masked_sequence(sequence_length: int, tokenizer, batch_size: int = 1, device: str = 'cuda'):
    """Create a fully masked sequence for generation."""
    seq = [tokenizer.mask_token] * sequence_length
    sequences = [''.join(seq)] * batch_size
    
    encoded = tokenizer(
        sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt'
    )
    return encoded['input_ids'].to(device)

def save_sequences_to_fasta(sequences: list, seq_len: int, save_path: str):
    """Save generated sequences to FASTA format."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as fp:
        for idx, seq in enumerate(sequences):
            fp.write(f">SEQUENCE_{idx}_L={seq_len}\n")
            fp.write(f"{seq}\n")

def generate_sequences(
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    planner_name: str = None,
    num_seqs: int = 100,
    seq_len: int = 128,
    num_steps: int = 128,
    temperature: float = 1.0,
    eta: float = 1.0,
    seed: int = None,
    device: str = 'cuda',
    save_dir: str = 'generation-results',
) -> tuple[list, float]:
    """Generate protein sequences using P2 sampling."""
    seed_everything(seed)
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model = model.eval().to(device)
    
    model_wrapper = ModelWrapper(model, tokenizer)
    
    # Load planner if specified
    planner = None
    if planner_name:
        print(f"Loading planner model {planner_name}...")
        planner_tokenizer = AutoTokenizer.from_pretrained(planner_name)
        planner_model = AutoModel.from_pretrained(planner_name)
        planner_model = planner_model.eval().to(device)
        planner = ModelWrapper(planner_model, planner_tokenizer)
    
    print("Creating initial sequence...")
    xt = create_masked_sequence(
        sequence_length=seq_len,
        tokenizer=tokenizer,
        batch_size=num_seqs,
        device=device
    )
    print(f"Initial sequence shape: {xt.shape}")
    
    print("Starting P2 sampling...")
    start_time = time.time()
    sampled_xt = p2_sampling(
        xt=xt,
        model=model_wrapper,
        mask_id=tokenizer.mask_token_id,
        num_steps=num_steps,
        tau=temperature,
        eta=eta,
        planner=planner,
    )
    
    elapsed_time = time.time() - start_time
    
    decoded_seqs = tokenizer.batch_decode(sampled_xt, skip_special_tokens=True)
    decoded_seqs = [''.join(seq.split()) for seq in decoded_seqs]
    
    # Save sequences
    save_path = os.path.join(save_dir, f"L_{seq_len}.fasta")
    save_sequences_to_fasta(decoded_seqs, seq_len, save_path)
    print(f"Saved sequences to {save_path}")
    
    return decoded_seqs, elapsed_time

def parse_args():
    parser = argparse.ArgumentParser(description="Protein Sequence Generation using P2 Sampling")
    parser.add_argument('--model_name', type=str, default="airkingbd/dplm_650m",)
    parser.add_argument('--planner_name', type=str, default=None,)
    parser.add_argument('--num_seqs', type=int, default=200,)
    parser.add_argument('--seq_len', type=int, default=200,)
    parser.add_argument('--num_steps', type=int, default=200,)
    parser.add_argument('--temperature', type=float, default=1.0,
                      help="Sampling temperature")
    parser.add_argument('--eta', type=float, default=1.0,
                      help="Stochasticity strength (0: deterministic, 1: default, >1: more stochastic)")
    parser.add_argument('--seed', type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument('--save_dir', type=str, default='generation-results',
                      help="Directory to save generated sequences")
    parser.add_argument('--esmfold_eval', action='store_true', default=False,
                      help="Run ESMFold evaluation")
    parser.add_argument('--max_tokens_per_batch', type=int, default=1024,
                      help="Maximum tokens per batch for ESMFold evaluation")
    parser.add_argument('--num_recycles', type=int, default=None,
                      help="Number of recycles for ESMFold")
    parser.add_argument('--cpu_only', action='store_true',
                      help="Use CPU only for ESMFold")
    parser.add_argument('--cpu_offload', action='store_true',
                      help="Enable CPU offloading for ESMFold")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\nProtein Sequence Generation Parameters:")
    print(f"Model: {args.model_name}")
    print(f"Planner: {args.planner_name if args.planner_name else 'None'}")
    print(f"Number of Sequences: {args.num_seqs}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Number of Steps: {args.num_steps}")
    print(f"Temperature: {args.temperature}")
    print(f"Eta: {args.eta}")
    print(f"Seed: {args.seed}")
    print(f"Save Directory: {args.save_dir}")
    print(f"ESMFold Evaluation: {args.esmfold_eval}")
    
    sequences, elapsed_time = generate_sequences(
        model_name=args.model_name,
        planner_name=args.planner_name,
        num_seqs=args.num_seqs,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        temperature=args.temperature,
        eta=args.eta,
        seed=args.seed,
        save_dir=args.save_dir
    )
    
    print(f"\nGeneration completed in {elapsed_time:.2f} seconds")
    print(f"Tokens/second: {args.num_seqs * args.seq_len / elapsed_time:.2f}")
    
    if args.esmfold_eval:
        print("\nRunning ESMFold evaluation...")
        save_dir = Path(args.save_dir)
        df = run_esmfold_eval(
            fasta_dir=save_dir,
            output_dir=save_dir / "esmfold_pdb",
            num_recycles=args.num_recycles,
            max_tokens_per_batch=args.max_tokens_per_batch,
            cpu_only=args.cpu_only,
            cpu_offload=args.cpu_offload
        )
        
        if not df.empty:
            # Add generation metadata
            df['Model'] = args.model_name
            df['Planner'] = args.planner_name if args.planner_name else "None"
            df['Temperature'] = args.temperature
            df['Eta'] = args.eta
            df['Steps'] = args.num_steps
            df['Generation Time'] = elapsed_time
            
            # Save results
            results_path = save_dir / "esmfold_results.csv"
            df.to_csv(results_path, index=False)
            print(f"\nSaved ESMFold results to {results_path}")
            
            # Calculate foldability
            foldable_count = df[
                (df['pLDDT'] > 80) & (df['pTM'] > 0.7) & (df['pAE'] < 10)
            ].shape[0]
            foldability = (foldable_count / len(df)) * 100
            print(f"Foldability: {foldability:.2f}%")
    
    print("\nSample sequences:")
    for i, seq in enumerate(sequences[:5]):
        print(f"Sequence {i+1}: {seq}")

if __name__ == '__main__':
    main() 