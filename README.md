# Path Planning

A Python package for sequence generation using P2 (Path Planning) sampling.

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/path_planning.git
cd path_planning

# Install the package
pip install -e .
```

### With ESMFold Support

To use the ESMFold evaluation functionality in the protein examples, install the package with the additional dependencies:

```bash
# Install the package with ESMFold support
pip install -e ".[protein]"
```

Alternatively, you can install the ESMFold dependencies manually:

```bash
# Install ESMFold and its dependencies
pip install "fair-esm[esmfold]"

# Install OpenFold and its remaining dependencies
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

## Usage

### Protein Sequence Generation

```bash
# Basic generation
python examples/protein/generate.py --num_seqs 10 --seq_len 128

# With ESMFold evaluation
python examples/protein/generate.py --num_seqs 10 --seq_len 128 --esmfold_eval --save_dir results/test_run

# With ESMFold options
python examples/protein/generate.py --num_seqs 10 --esmfold_eval --max_tokens_per_batch 512 --num_recycles 4

# CPU options for ESMFold
python examples/protein/generate.py --esmfold_eval --cpu_only  # or --cpu_offload
```

### Command-line Arguments

- `--model_name`: ESM2 model name (default: "facebook/esm2_t6_8M_UR50D")
- `--num_seqs`: Number of sequences to generate (default: 100)
- `--seq_len`: Length of sequences to generate (default: 128)
- `--num_steps`: Number of P2 sampling steps (default: 128)
- `--temperature`: Sampling temperature (default: 1.0)
- `--eta`: Stochasticity strength (0: deterministic, 1: default, >1: more stochastic) (default: 1.0)
- `--seed`: Random seed for reproducibility (default: 42)
- `--save_dir`: Directory to save generated sequences (default: 'generation-results')
- `--esmfold_eval`: Run ESMFold evaluation (default: False)
- `--max_tokens_per_batch`: Maximum tokens per batch for ESMFold evaluation (default: 1024)
- `--num_recycles`: Number of recycles for ESMFold (default: None)
- `--cpu_only`: Use CPU only for ESMFold (default: False)
- `--cpu_offload`: Enable CPU offloading for ESMFold (default: False)

## API Usage

```python
from path_planning import p2_sampling, seed_everything

# Set random seed for reproducibility
seed_everything(42)

# Use P2 sampling in your code
sampled_sequence = p2_sampling(
    xt=initial_sequence,
    model=model_wrapper,
    tokenizer=tokenizer,
    num_steps=128,
    tau=1.0,
    eta=1.0,
    score_type='confidence'
)
``` 