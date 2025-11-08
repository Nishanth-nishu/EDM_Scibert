import selfies as sf
import glob
from collections import Counter

# Path to your SELFIES dataset folder
dataset_dir = "../../datasets/SELFIES"

# Collect all files that contain SELFIES strings
files = glob.glob(f"{dataset_dir}/*selfies.txt")

vocab_counter = Counter()

print(f"Scanning {len(files)} files for SELFIES tokens...")

for f in files:
    with open(f) as r:
        for line in r:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            selfie = parts[1]
            tokens = sf.split_selfies(selfie)
            vocab_counter.update(tokens)

# Sort by frequency
sorted_vocab = [tok for tok, _ in vocab_counter.most_common()]

# Save
output_path = f"{dataset_dir}/selfies_vocab_full.txt"
with open(output_path, "w") as w:
    for tok in sorted_vocab:
        w.write(tok + "\n")

print(f"\n✓ Saved full SELFIES vocab to: {output_path}")
print(f"✓ Total unique tokens: {len(sorted_vocab)}")
