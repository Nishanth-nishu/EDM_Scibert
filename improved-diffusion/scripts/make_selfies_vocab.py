import selfies as sf
from collections import Counter
import os

dataset_dir = "../../datasets/SELFIES"
output_vocab = os.path.join(dataset_dir, "selfies_vocab_full.txt")

files = [
    os.path.join(dataset_dir, "train.txt"),
    os.path.join(dataset_dir, "validation.txt"),
    os.path.join(dataset_dir, "test.txt"),
    os.path.join(dataset_dir, "mini.txt"),
]

tokens = Counter()
n_total, n_valid = 0, 0

print("\n" + "="*60)
print("Building SELFIES Vocabulary from tab-separated dataset files")
print("="*60 + "\n")

for f in files:
    if not os.path.exists(f):
        print(f"⚠️ Skipping missing file: {f}")
        continue

    print(f"Processing {f}...")
    with open(f) as fin:
        for line in fin:
            n_total += 1
            line = line.strip()
            if not line or line.startswith("CID"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            raw_mol = parts[1].strip()  # second column = SELFIES or SMILES

            # Skip placeholders like '*'
            if raw_mol in ["*", "", "NA", "N/A"]:
                continue

            try:
                # If it’s already SELFIES, it will start with '['
                if raw_mol.startswith("["):
                    selfies_str = raw_mol
                else:
                    selfies_str = sf.encoder(raw_mol)

                # Count tokens
                for token in sf.split_selfies(selfies_str):
                    tokens[token] += 1
                n_valid += 1

            except Exception:
                continue

specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
vocab = specials + list(tokens.keys())

os.makedirs(dataset_dir, exist_ok=True)
with open(output_vocab, "w") as fout:
    for tok in vocab:
        fout.write(tok + "\n")

print("\n" + "="*60)
print(f"✅ Saved SELFIES vocab to {output_vocab}")
print(f"✅ Total valid molecules processed: {n_valid}/{n_total}")
print(f"✅ Total unique tokens: {len(vocab)}")
print("="*60 + "\n")

print("Most common tokens:")
for t, c in tokens.most_common(10):
    print(f"  {t}: {c}")
