"""
Convert SMILES datasets to SELFIES format
Builds SELFIES vocabulary from the dataset
"""

import os
import selfies as sf
from collections import Counter
from tqdm import tqdm


def convert_dataset_file(input_path, output_path, collect_tokens=False):
    """
    Convert a single dataset file from SMILES to SELFIES
    
    Args:
        input_path: Path to SMILES dataset file
        output_path: Path to save SELFIES dataset file
        collect_tokens: If True, only collect tokens without writing output
    
    Returns:
        Set of all SELFIES tokens encountered
    """
    all_tokens = set()
    token_counts = Counter()
    converted_count = 0
    failed_count = 0
    
    if not collect_tokens:
        print(f"Converting {input_path} → {output_path}")
    
    f_out = None if collect_tokens else open(output_path, 'w')
    
    try:
        with open(input_path, 'r') as f_in:
            for i, line in enumerate(tqdm(f_in, disable=collect_tokens)):
                if i == 0:
                    # Write header
                    if f_out:
                        f_out.write(line)
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    print(f"Skipping malformed line {i}: {line[:50]}")
                    continue
                
                cid, smiles, description = parts
                
                # Skip invalid SMILES
                if smiles == '*':
                    if f_out:
                        f_out.write(f"{cid}\t*\t{description}\n")
                    continue
                
                # Convert SMILES to SELFIES
                try:
                    selfies = sf.encoder(smiles)
                    
                    # Collect tokens for vocabulary
                    tokens = list(sf.split_selfies(selfies))
                    all_tokens.update(tokens)
                    token_counts.update(tokens)
                    
                    # Write converted line
                    if f_out:
                        f_out.write(f"{cid}\t{selfies}\t{description}\n")
                    converted_count += 1
                    
                except Exception as e:
                    # print(f"Failed to convert SMILES at line {i}: {smiles}")
                    # print(f"Error: {e}")
                    # Write original SMILES with marker
                    if f_out:
                        f_out.write(f"{cid}\t*\t{description}\n")
                    failed_count += 1
    finally:
        if f_out:
            f_out.close()
    
    if not collect_tokens:
        print(f"✓ Converted: {converted_count}")
        print(f"✗ Failed: {failed_count}")
        print(f"Unique tokens: {len(all_tokens)}")
    
    return all_tokens, token_counts


def build_selfies_vocabulary(dataset_dir, output_vocab_path):
    """
    Build SELFIES vocabulary from all dataset files
    
    Args:
        dataset_dir: Directory containing SMILES dataset files
        output_vocab_path: Path to save vocabulary file
    
    Returns:
        List of SELFIES tokens
    """
    print("\n" + "="*60)
    print("Building SELFIES Vocabulary")
    print("="*60)
    
    all_tokens = set()
    token_counts = Counter()
    
    # Process all dataset splits
    splits = ['train.txt', 'validation.txt', 'test.txt', 'mini.txt']
    
    # Check for additional splits
    if os.path.exists(os.path.join(dataset_dir, 'train_val_256.txt')):
        splits.append('train_val_256.txt')
    
    for split in splits:
        input_path = os.path.join(dataset_dir, split)
        if not os.path.exists(input_path):
            print(f"Skipping {split} (not found)")
            continue
        
        print(f"\nCollecting tokens from {split}...")
        
        with open(input_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc=f"Processing {split}")):
                if i == 0:
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) != 3 or parts[1] == '*':
                    continue
                
                smiles = parts[1]
                
                try:
                    selfies = sf.encoder(smiles)
                    tokens = list(sf.split_selfies(selfies))
                    all_tokens.update(tokens)
                    token_counts.update(tokens)
                except:
                    continue
    
    # Sort tokens by frequency (most common first)
    sorted_tokens = [token for token, _ in token_counts.most_common()]
    
    # Write vocabulary
    with open(output_vocab_path, 'w') as f:
        for token in sorted_tokens:
            f.write(f"{token}\n")
    
    print(f"\n✓ Vocabulary saved to {output_vocab_path}")
    print(f"✓ Total unique tokens: {len(sorted_tokens)}")
    print(f"\nTop 30 most common tokens:")
    for token, count in token_counts.most_common(30):
        print(f"  {token}: {count}")
    
    return sorted_tokens, token_counts


def main():
    """Main conversion pipeline"""
    
    # Paths
    smiles_dir = '../../datasets/SMILES'
    selfies_dir = '../../datasets/SELFIES'
    
    # Create SELFIES directory
    os.makedirs(selfies_dir, exist_ok=True)
    
    print("="*60)
    print("SMILES → SELFIES Dataset Conversion")
    print("="*60)
    
    # Step 1: Build vocabulary FIRST (before conversion)
    print("\nStep 1: Building vocabulary from SMILES dataset...")
    vocab_path = os.path.join(selfies_dir, 'selfies_vocab.txt')
    
    vocabulary, token_counts = build_selfies_vocabulary(smiles_dir, vocab_path)
    
    # Step 2: Convert dataset files
    print("\n" + "="*60)
    print("Step 2: Converting dataset files...")
    print("="*60)
    
    splits = {
        'train.txt': 'train.txt',
        'validation.txt': 'validation.txt',
        'test.txt': 'test.txt',
        'mini.txt': 'mini.txt',
        'train_val_256.txt': 'train_val_256.txt'
    }
    
    for input_file, output_file in splits.items():
        input_path = os.path.join(smiles_dir, input_file)
        output_path = os.path.join(selfies_dir, output_file)
        
        if not os.path.exists(input_path):
            print(f"\nSkipping {input_file} (not found)")
            continue
        
        print(f"\n{'-'*60}")
        tokens, _ = convert_dataset_file(input_path, output_path, collect_tokens=False)
    
    print("\n" + "="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    print(f"Converted datasets saved to: {selfies_dir}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    # Step 3: Verify conversion with sample
    print("\n" + "="*60)
    print("Verification Sample:")
    print("="*60)
    
    sample_file = os.path.join(selfies_dir, 'mini.txt')
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                # Show first 3 samples
                for idx in range(1, min(4, len(lines))):
                    parts = lines[idx].strip().split('\t')
                    if len(parts) == 3:
                        cid, selfies, desc = parts
                        if selfies != '*':
                            try:
                                smiles = sf.decoder(selfies)
                                print(f"\nSample {idx}:")
                                print(f"  CID: {cid}")
                                print(f"  SELFIES: {selfies[:60]}...")
                                print(f"  SMILES: {smiles}")
                                print(f"  Tokens: {len(list(sf.split_selfies(selfies)))}")
                            except Exception as e:
                                print(f"\nWarning: Failed to decode SELFIES for CID {cid}")
                                print(f"  Error: {e}")
    
    # Step 4: Statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    
    for split_file in ['train.txt', 'validation.txt', 'test.txt', 'mini.txt', 'train_val_256.txt']:
        split_path = os.path.join(selfies_dir, split_file)
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                lines = f.readlines()
                valid_count = sum(1 for line in lines[1:] if line.strip().split('\t')[1] != '*')
                print(f"  {split_file:20s}: {valid_count:6d} valid samples")


if __name__ == "__main__":
    main()