import selfies as sf
from collections import Counter

files = [
    '../../datasets/SELFIES/train.txt',
    '../../datasets/SELFIES/validation.txt',
    '../../datasets/SELFIES/test.txt',
]

tokens = Counter()

for f in files:
    with open(f) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # encode the molecule into SELFIES (if not already SELFIES)
            try:
                s = sf.encoder(line) if not line.startswith('[') else line
                # tokenize SELFIES string into symbols
                for token in sf.split_selfies(s):
                    tokens[token] += 1
            except Exception:
                pass

# Add special tokens
specials = ['<pad>', '<bos>', '<eos>', '<unk>']
vocab = specials + list(tokens.keys())

with open('../../datasets/SELFIES/selfies_vocab_full.txt', 'w') as fout:
    for tok in vocab:
        fout.write(tok + '\n')

print(f"✅ Created full vocab with {len(vocab)} tokens")
