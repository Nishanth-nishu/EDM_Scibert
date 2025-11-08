"""
Clean SELFIES dataset by removing invalid entries
"""

import os
import selfies as sf
from tqdm import tqdm

def clean_dataset_file(input_path, output_path):
    """
    Clean a SELFIES dataset file by validating each entry
    
    Args:
        input_path: Path to input dataset file
        output_path: Path to save cleaned dataset
    
    Returns:
        Statistics about cleaning
    """
    print(f"\nCleaning: {os.path.basename(input_path)}")
    print("-" * 60)
    
    total_lines = 0
    valid_lines = 0
    invalid_selfies = []
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for i, line in enumerate(tqdm(f_in)):
            if i == 0:
                # Write header
                f_out.write(line)
                continue
            
            total_lines += 1
            parts = line.strip().split('\t')
            
            if len(parts) != 3:
                print(f"Skipping malformed line {i+1}")
                continue
            
            cid, selfies, desc = parts
            
            # Keep entries marked as invalid
            if selfies == '*':
                f_out.write(line)
                valid_lines += 1
                continue
            
            # Validate SELFIES
            try:
                tokens = list(sf.split_selfies(selfies))
                
                # Must have at least one token
                if len(tokens) == 0:
                    invalid_selfies.append((i+1, cid, selfies[:50], "Empty token list"))
                    continue
                
                # Try to decode to verify it's valid
                smiles = sf.decoder(selfies)
                
                # If we get here, it's valid
                f_out.write(line)
                valid_lines += 1
                
            except Exception as e:
                invalid_selfies.append((i+1, cid, selfies[:50], str(e)))
    
    print(f"✓ Total lines: {total_lines}")
    print(f"✓ Valid lines: {valid_lines}")
    print(f"✗ Invalid lines: {len(invalid_selfies)}")
    
    if invalid_selfies:
        print(f"\nFirst 10 invalid entries:")
        for line_num, cid, selfies, error in invalid_selfies[:10]:
            print(f"  Line {line_num}, CID {cid}: {selfies}...")
            print(f"    Error: {error}")
    
    return {
        'total': total_lines,
        'valid': valid_lines,
        'invalid': len(invalid_selfies),
        'invalid_entries': invalid_selfies
    }

def main():
    """Main cleaning function"""
    
    print("="*60)
    print("SELFIES Dataset Cleaning Tool")
    print("="*60)
    
    dataset_dir = '../../datasets/SELFIES'
    
    splits = ['train.txt', 'validation.txt', 'test.txt', 'mini.txt', 'train_val_256.txt']
    
    all_stats = {}
    
    for split in splits:
        input_path = os.path.join(dataset_dir, split)
        
        if not os.path.exists(input_path):
            print(f"\nSkipping {split} (not found)")
            continue
        
        # Create backup
        backup_path = input_path + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(input_path, backup_path)
            print(f"Created backup: {backup_path}")
        
        # Clean to temporary file first
        temp_path = input_path + '.cleaned'
        stats = clean_dataset_file(input_path, temp_path)
        all_stats[split] = stats
        
        # Replace original with cleaned version
        os.replace(temp_path, input_path)
        print(f"✓ Updated {split}")
    
    # Summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    
    total_all = sum(s['total'] for s in all_stats.values())
    valid_all = sum(s['valid'] for s in all_stats.values())
    invalid_all = sum(s['invalid'] for s in all_stats.values())
    
    print(f"\nOverall Statistics:")
    print(f"  Total entries: {total_all}")
    print(f"  Valid entries: {valid_all} ({100*valid_all/total_all:.2f}%)")
    print(f"  Invalid entries removed: {invalid_all} ({100*invalid_all/total_all:.2f}%)")
    
    print("\nPer-split breakdown:")
    for split, stats in all_stats.items():
        validity = 100 * stats['valid'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {split:20s}: {stats['valid']:6d}/{stats['total']:6d} ({validity:.1f}%)")
    
    if invalid_all > 0:
        print("\n⚠ Some invalid entries were removed.")
        print("  Original files backed up with .backup extension")
        print("  You may need to regenerate description embeddings:")
        print("\n    python process_text_selfies.py -i train_val_256")
        print("    python process_text_selfies.py -i validation")
        print("    python process_text_selfies.py -i test")
    else:
        print("\n✓ All datasets are clean!")

if __name__ == "__main__":
    main()
