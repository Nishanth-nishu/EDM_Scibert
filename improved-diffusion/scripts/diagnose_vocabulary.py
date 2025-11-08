"""
Diagnostic script to check vocabulary and dataset consistency
"""

import os
import selfies as sf
from collections import Counter

def check_vocabulary(vocab_path):
    """Check vocabulary file"""
    print("="*60)
    print("Checking Vocabulary File")
    print("="*60)
    
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary file not found: {vocab_path}")
        return None
    
    with open(vocab_path, 'r') as f:
        tokens = [line.strip() for line in f.readlines()]
    
    print(f"✓ Vocabulary file exists")
    print(f"✓ Total tokens: {len(tokens)}")
    print(f"\nFirst 20 tokens:")
    for i, token in enumerate(tokens[:20]):
        print(f"  {i+1:3d}. {token}")
    
    return set(tokens)

def check_dataset(dataset_path, vocab_tokens=None):
    """Check dataset and find tokens not in vocabulary"""
    print("\n" + "="*60)
    print(f"Checking Dataset: {os.path.basename(dataset_path)}")
    print("="*60)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return
    
    all_tokens = Counter()
    missing_tokens = Counter()
    total_samples = 0
    valid_samples = 0
    
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            total_samples += 1
            cid, selfies, desc = parts
            
            if selfies == '*':
                continue
            
            valid_samples += 1
            
            # Split SELFIES and count tokens
            try:
                tokens = list(sf.split_selfies(selfies))
                all_tokens.update(tokens)
                
                # Check for missing tokens
                if vocab_tokens:
                    for token in tokens:
                        if token not in vocab_tokens:
                            missing_tokens[token] += 1
            except Exception as e:
                print(f"Warning: Failed to parse SELFIES at line {i+1}")
                print(f"  SELFIES: {selfies[:80]}")
                print(f"  Error: {e}")
    
    print(f"✓ Total samples: {total_samples}")
    print(f"✓ Valid samples: {valid_samples}")
    print(f"✓ Unique tokens in dataset: {len(all_tokens)}")
    
    if missing_tokens:
        print(f"\n❌ Found {len(missing_tokens)} tokens NOT in vocabulary!")
        print(f"\nTop 20 missing tokens:")
        for token, count in missing_tokens.most_common(20):
            print(f"  {token}: {count} occurrences")
    else:
        print(f"\n✓ All tokens are in vocabulary!")
    
    print(f"\nTop 20 most common tokens in dataset:")
    for token, count in all_tokens.most_common(20):
        status = "✓" if (vocab_tokens is None or token in vocab_tokens) else "❌"
        print(f"  {status} {token}: {count}")
    
    return all_tokens, missing_tokens

def main():
    """Main diagnostic function"""
    
    print("\n" + "="*70)
    print("SELFIES VOCABULARY DIAGNOSTIC TOOL")
    print("="*70)
    
    # Paths
    vocab_path = '../../datasets/SELFIES/selfies_vocab.txt'
    dataset_dir = '../../datasets/SELFIES'
    
    # Check vocabulary
    vocab_tokens = check_vocabulary(vocab_path)
    
    # Check each dataset split
    splits = ['train.txt', 'validation.txt', 'test.txt', 'mini.txt', 'train_val_256.txt']
    
    all_missing = Counter()
    
    for split in splits:
        dataset_path = os.path.join(dataset_dir, split)
        if os.path.exists(dataset_path):
            dataset_tokens, missing_tokens = check_dataset(dataset_path, vocab_tokens)
            all_missing.update(missing_tokens)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_missing:
        print(f"❌ PROBLEM DETECTED!")
        print(f"   Found {len(all_missing)} unique missing tokens across all datasets")
        print(f"\n   All missing tokens:")
        for token, count in all_missing.most_common():
            print(f"     {token}: {count} occurrences")
        
        print("\n" + "="*60)
        print("SOLUTION:")
        print("="*60)
        print("The vocabulary file is incomplete or was built incorrectly.")
        print("Please run the dataset conversion script again:")
        print("\n  python convert_smiles_to_selfies.py")
        print("\nThis will rebuild the vocabulary from scratch.")
        
    else:
        print("✓ No issues found!")
        print("✓ All dataset tokens are in the vocabulary")
        print("\nYou can proceed with training.")

if __name__ == "__main__":
    main()
