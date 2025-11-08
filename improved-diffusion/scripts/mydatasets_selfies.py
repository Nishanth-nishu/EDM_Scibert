"""
Dataset loader for SELFIES molecular representations
Modified from original SMILES-based loader
"""

from torch.utils.data import DataLoader, Dataset
import torch
import random
import selfies as sf
from torch.utils.data import DistributedSampler


def get_dataloader(dataset, batchsize, rank, world_size):
    """Create distributed dataloader with cycling"""
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    def collate(batch):
        toked_selfies = [i['tok_selfies'] for i in batch]
        desc_states = [i['desc_state'] for i in batch]
        desc_mask = [i['desc_mask'] for i in batch]
        corrupted_toked_selfies = [i['corrupted_toked_selfies'] for i in batch]
        
        return (
            torch.cat(toked_selfies, dim=0),
            torch.cat(desc_states, dim=0),
            torch.cat(desc_mask, dim=0),
            torch.cat(corrupted_toked_selfies, dim=0)
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate,
        sampler=sampler
    )
    
    def cycle():
        """Infinite cycling through epochs"""
        ec = 0
        while True:
            dataloader.sampler.set_epoch(ec)
            for batch in dataloader:
                yield batch
            ec += 1
    
    return iter(cycle())


class ChEBIdataset(Dataset):
    """
    ChEBI dataset loader for SELFIES representations
    """
    
    def __init__(
        self, 
        dir, 
        selfies_tokenizer, 
        split, 
        replace_desc=False, 
        pre=None, 
        prob=0, 
        load_state=True, 
        corrupt_prob=0.4, 
        mask_desc=False
    ):
        """
        Args:
            dir: Dataset directory (should contain SELFIES files)
            selfies_tokenizer: SELFIES tokenizer instance
            split: Dataset split ('train', 'test', 'validation', 'mini', 'train_val_256')
            replace_desc: Whether to replace description format
            pre: Prefix for descriptions
            prob: Probability of applying augmentation
            load_state: Whether to load pre-computed description states
            corrupt_prob: Probability of corrupting SELFIES during training
            mask_desc: Whether to mask descriptions
        """
        super().__init__()
        
        self.dir = dir
        self.selfies_tokenizer = selfies_tokenizer
        self.split = split
        self.replace_desc = replace_desc
        self.pre = pre
        self.prob = prob
        self.corrupt_prob = corrupt_prob
        self.mask_desc = mask_desc
        
        print(f'Corruption probability: {self.corrupt_prob}')
        print(f'Mask descriptions: {self.mask_desc}')
        
        assert split in ['train', 'test', 'validation', 'mini', 'train_val_256'], \
            f"Invalid split: {split}"
        
        # Load dataset
        self.ori_data = self.get_ori_data()
        print(f'Loaded {len(self.ori_data)} samples from {split} split')
        
        # Load pre-computed description states
        self.load_state = load_state
        if load_state:
            self.desc_state = self.get_desc_state()
    
    def get_desc_state(self):
        """Load pre-computed description embeddings"""
        import os.path as osp
        file_path = osp.join(self.dir, self.split + '_desc_states.pt')
        
        try:
            desc_states = torch.load(file_path)
            print(f'Loaded description states from {file_path}')
            return desc_states
        except FileNotFoundError:
            print(f'Warning: Description states not found at {file_path}')
            print('Please run process_text.py to generate description embeddings')
            raise
    
    def get_ori_data(self):
        """Load SELFIES dataset"""
        import os.path as osp
        
        if self.replace_desc:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        
        res = []
        file_path = osp.join(self.dir, self.split + '.txt')
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # Skip header
                    continue
                
                parts = line.split('\t')
                assert len(parts) == 3, f"Invalid line format at line {i}"
                
                cid, selfies, desc = parts
                
                # Skip invalid entries
                if selfies.strip() == '*':
                    continue
                
                desc = desc.strip()
                
                # Optionally reformat description
                if self.replace_desc:
                    doc = nlp(desc)
                    for token in doc:
                        if token.text == 'is':
                            desc = 'The molecule ' + desc[token.idx:]
                            break
                
                res.append((int(cid), selfies.strip(), desc))
        
        return res
    
    def augment_selfies(self, selfies_str):
        """
        Augment SELFIES representation
        SELFIES support randomized encoding for augmentation
        """
        p = random.random()
        
        if p < self.prob:
            try:
                # Convert to SMILES and back with randomization
                smiles = sf.decoder(selfies_str)
                
                # Randomized encoding (if available in selfies version)
                # This provides equivalent molecules with different SELFIES representations
                augmented_selfies = sf.encoder(smiles)
                
                return augmented_selfies
            except:
                return selfies_str
        
        return selfies_str
    
    def __len__(self):
        return len(self.ori_data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        cid, selfies, desc = self.ori_data[idx]
        
        # Apply augmentation
        selfies = self.augment_selfies(selfies)
        
        # Tokenize SELFIES
        tok_selfies = self.selfies_tokenizer(selfies)
        
        # Apply corruption for training
        if random.random() < self.corrupt_prob:
            corrupted_toked_selfies = self.selfies_tokenizer.corrupt(selfies)
        else:
            corrupted_toked_selfies = tok_selfies
        
        # Build sample dictionary
        sample = {
            'cid': cid,
            'selfies': selfies,
            'desc': desc,
            'tok_selfies': tok_selfies,
            'corrupted_toked_selfies': corrupted_toked_selfies,
            'tok_desc': None,
            'dec_mask': None
        }
        
        # Add description embeddings
        if self.load_state:
            sample['desc_state'] = self.desc_state[cid]['states']
            sample['desc_mask'] = self.desc_state[cid]['mask']
            
            # Optionally mask descriptions
            if self.mask_desc:
                sample['desc_state'] = torch.zeros_like(sample['desc_state'])
                sample['desc_mask'] = torch.ones_like(sample['desc_mask'])
        
        return sample


# Backward compatibility
def changeorder(selfies, shuffle):
    """
    For SELFIES, augmentation is handled differently
    This function is kept for compatibility but not used
    """
    if shuffle:
        print("Warning: Direct reordering not applicable to SELFIES")
    return selfies
