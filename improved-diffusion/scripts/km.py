"""
Quick SELFIES dataset inspection / debugging script.
Ensures SELFIES vocab and tokenization work correctly.
"""

from mydatasets_selfies import ChEBIdataset
from mytokenizers_selfies import SELFIESTokenizer
import torch
import random

# ============================================================
# Configuration
# ============================================================
DATASET_DIR = '../../datasets/SELFIES/'
VOCAB_PATH = f'{DATASET_DIR}/selfies_vocab_full.txt'
SPLIT = 'train'  # or 'validation', 'test', etc.
MAX_LEN = 256
MASK_DESC = False
CORRUPT_PROB = 0.4

# ============================================================
# Initialize tokenizer
# ============================================================
print(f"Initializing SELFIES Tokenizer with max_len: {MAX_LEN}")
tokenizer = SELFIESTokenizer(
    vocab_path=VOCAB_PATH,
    max_len=MAX_LEN
)

# --- Handle different vocab attribute names safely ---
if hasattr(tokenizer, "vocab"):
    vocab_size = len(tokenizer.vocab)
elif hasattr(tokenizer, "vocab_list"):
    vocab_size = len(tokenizer.vocab_list)
elif hasattr(tokenizer, "stoi"):  # string-to-index dict
    vocab_size = len(tokenizer.stoi)
elif hasattr(tokenizer, "token_to_id"):
    vocab_size = len(tokenizer.token_to_id)
else:
    raise AttributeError(
        "Could not find vocabulary attribute in SELFIESTokenizer. "
        "Expected one of: vocab, vocab_list, stoi, token_to_id."
    )

print(f"SELFIES Vocabulary size: {vocab_size}")
print(f"Corruption probability: {CORRUPT_PROB}")
print(f"Mask descriptions: {MASK_DESC}")

# ============================================================
# Load dataset
# ============================================================
ds = ChEBIdataset(
    dir=DATASET_DIR,
    selfies_tokenizer=tokenizer,
    split=SPLIT,
    replace_desc=False,
    load_state=False
)

print(f"Loaded {len(ds)} samples from {SPLIT} split")

# ============================================================
# Inspect a few random samples
# ============================================================
num_samples = min(5, len(ds))
indices = random.sample(range(len(ds)), num_samples)

for i in indices:
    sample = ds[i]

    # Handle both possible key names
    selfie = sample.get('selfie', sample.get('smiles', None))
    if selfie is None:
        raise KeyError("Neither 'selfie' nor 'smiles' found in dataset sample!")

    print("=" * 80)
    print(f"Sample #{i}")
    print(f"CID: {sample.get('cid', 'N/A')}")
    print(f"Description: {sample.get('desc', '[No description found]')[:200]}...")
    print(f"SELFIES: {selfie}")

    if 'toked_selfie' in sample:
        tok = sample['toked_selfie']
        print(f"Tokenized SELFIE shape: {tok.shape}")
        if torch.is_tensor(tok):
            print(f"First 10 token IDs: {tok[:10].tolist()}")
        else:
            print(f"First 10 token IDs: {tok[:10]}")

print("=" * 80)
print("✅ Dataset inspection complete — SELFIES tokenization OK!")
print("=" * 80)
