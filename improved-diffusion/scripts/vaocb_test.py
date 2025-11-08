from mytokenizers_selfies import SELFIESTokenizer
from mydatasets_selfies import ChEBIdataset

tok = SELFIESTokenizer(vocab_path='../../datasets/SELFIES/selfies_vocab_full.txt')
ds = ChEBIdataset('../../datasets/SELFIES/', tok, 'train', replace_desc=False, load_state=False)

unk_count = 0
for i in range(1000):  # sample a few molecules
    selfie = ds[i]['selfie']
    toks = tok.tokenize(selfie)
    unk_count += toks.count(tok.unk_token)

print(f"Unknown tokens in first 1000 samples: {unk_count}")
