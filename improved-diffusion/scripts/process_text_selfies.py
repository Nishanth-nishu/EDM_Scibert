"""
Process text descriptions and generate embeddings for SELFIES dataset.
Modified from original SMILES version.
"""

from mydatasets_selfies import ChEBIdataset
import torch
import transformers
from mytokenizers_selfies import SELFIESTokenizer
from transformers import AutoModel, AutoTokenizer
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Dataset split to process")
args = parser.parse_args()
split = args.input

print("="*60)
print(f"Processing descriptions for split: {split}")
print("="*60)

# Initialize SELFIES tokenizer (needed for dataset loading)
selfies_tokenizer = SELFIESTokenizer(
    vocab_path='../../datasets/SELFIES/selfies_vocab.txt'
)

# Load dataset without pre-computed states and WITHOUT corruption
# We only need descriptions, not corrupted SELFIES
train_dataset = ChEBIdataset(
    dir='../../datasets/SELFIES/',
    selfies_tokenizer=selfies_tokenizer,
    split=split,
    replace_desc=False,
    load_state=False,
    corrupt_prob=0.0  # IMPORTANT: No corruption when processing descriptions
)

print(f"Dataset size: {len(train_dataset)}")

# Load SciBERT model for description encoding
print("\nLoading SciBERT model...")
model = AutoModel.from_pretrained('../../scibert')
tokenizer = AutoTokenizer.from_pretrained('../../scibert')

volume = {}

model = model.cuda()
model.eval()

print("\nProcessing descriptions...")
print("-"*60)

failed_samples = []

with torch.no_grad():
    for i in tqdm(range(len(train_dataset)), desc="Encoding descriptions"):
        try:
            sample = train_dataset[i]
            cid = sample['cid']
            desc = sample['desc']
            
            # Tokenize description
            tok_output = tokenizer(
                desc,
                max_length=216,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            toked_desc = tok_output['input_ids'].cuda()
            toked_desc_attention_mask = tok_output['attention_mask'].cuda()
            
            assert toked_desc.shape[1] == 216, f"Unexpected sequence length: {toked_desc.shape[1]}"
            
            # Get embeddings
            last_hidden = model(toked_desc).last_hidden_state
            
            # Store embeddings
            volume[cid] = {
                'states': last_hidden.cpu(),
                'mask': toked_desc_attention_mask.cpu()
            }
        
        except Exception as e:
            failed_samples.append((i, cid, str(e)))
            print(f"\nWarning: Failed to process sample {i}, CID {cid}")
            print(f"  Error: {e}")
            continue

# Save embeddings
output_path = f'../../datasets/SELFIES/{split}_desc_states.pt'
torch.save(volume, output_path)

print("\n" + "="*60)
print(f"✓ Saved description states to: {output_path}")
print(f"✓ Processed {len(volume)} samples")

if failed_samples:
    print(f"⚠ Failed to process {len(failed_samples)} samples")
    print("\nFailed samples:")
    for idx, cid, error in failed_samples[:10]:  # Show first 10
        print(f"  Sample {idx}, CID {cid}: {error}")

print("="*60)