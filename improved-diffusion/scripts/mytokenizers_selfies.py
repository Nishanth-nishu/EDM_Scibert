"""
SELFIES Tokenizer for Molecular Diffusion Models
Replaces SMILES tokenization with SELFIES encoding
"""

import torch
import random
import selfies as sf
from typing import List, Union


class SELFIESTokenizer:
    """
    Tokenizer for SELFIES molecular representations.
    Handles encoding/decoding and corruption for diffusion training.
    """
    
    def __init__(self, vocab_path='../../datasets/SELFIES/selfies_vocab.txt', max_len=256):
        """
        Args:
            vocab_path: Path to SELFIES vocabulary file
            max_len: Maximum sequence length (SELFIES are typically longer than SMILES)
        """
        print(f'Initializing SELFIES Tokenizer with max_len: {max_len}')
        
        self.max_len = max_len
        
        # Load or create vocabulary
        self.idtotok, self.toktoid = self._load_vocab(vocab_path)
        self.vocab_size = len(self.idtotok)
        
        # Special tokens
        self.PAD_ID = 0
        self.SOS_ID = 1
        self.EOS_ID = 2
        
        print(f'SELFIES Vocabulary size: {self.vocab_size}')
    
    def _load_vocab(self, vocab_path):
        """Load SELFIES vocabulary from file"""
        try:
            with open(vocab_path, 'r') as f:
                tokens = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Vocabulary file not found at {vocab_path}")
            print("Please run the vocabulary generation script first")
            raise
        
        # Build token mappings (reserve 0, 1, 2 for special tokens)
        idtotok = {
            0: '[PAD]',
            1: '[SOS]',
            2: '[EOS]',
            3: '[UNK]'  # Add unknown token
        }
        
        for idx, token in enumerate(tokens):
            idtotok[idx + 4] = token
        
        toktoid = {v: k for k, v in idtotok.items()}
        
        return idtotok, toktoid
    
    def smiles_to_selfies(self, smiles: str) -> str:
        """Convert SMILES to SELFIES"""
        try:
            selfies = sf.encoder(smiles)
            return selfies
        except Exception as e:
            print(f"Error converting SMILES to SELFIES: {smiles}")
            print(f"Error: {e}")
            return None
    
    def selfies_to_smiles(self, selfies: str) -> str:
        """Convert SELFIES back to SMILES"""
        try:
            smiles = sf.decoder(selfies)
            return smiles
        except Exception as e:
            print(f"Error converting SELFIES to SMILES: {selfies}")
            print(f"Error: {e}")
            return None
    
    def encode_one(self, selfies_str: str) -> torch.Tensor:
        """
        Encode a single SELFIES string to token IDs
        
        Args:
            selfies_str: SELFIES string (e.g., "[C][N][C]")
        
        Returns:
            Tensor of shape (1, max_len) with token IDs
        """
        # Split SELFIES into tokens
        tokens = list(sf.split_selfies(selfies_str))
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.toktoid:
                token_ids.append(self.toktoid[token])
            else:
                # Unknown token - skip or use a special UNK token
                print(f"Warning: Unknown SELFIES token: {token}")
                continue
        
        # Add SOS and EOS
        result = [self.SOS_ID] + token_ids + [self.EOS_ID]
        
        # Pad or truncate
        if len(result) < self.max_len:
            result += [self.PAD_ID] * (self.max_len - len(result))
        else:
            result = result[:self.max_len]
            result[-1] = self.EOS_ID
        
        return torch.LongTensor([result])
    
    def __call__(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode SELFIES string(s) to token IDs
        
        Args:
            inputs: Single SELFIES string or list of SELFIES strings
        
        Returns:
            Tensor of shape (batch_size, max_len)
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        tensors = []
        for selfies_str in inputs:
            tensors.append(self.encode_one(selfies_str))
        
        return torch.cat(tensors, dim=0)
    
    def decode_one(self, ids: torch.Tensor) -> str:
        """
        Decode token IDs to SELFIES string
        
        Args:
            ids: Tensor of token IDs
        
        Returns:
            SELFIES string
        """
        tokens = []
        for idx in ids:
            token_id = idx.item()
            if token_id == self.PAD_ID or token_id == self.EOS_ID:
                break
            if token_id == self.SOS_ID:
                continue
            
            if token_id in self.idtotok:
                tokens.append(self.idtotok[token_id])
        
        return ''.join(tokens)
    
    def decode(self, ids: torch.Tensor) -> List[str]:
        """
        Decode batch of token IDs to SELFIES strings
        
        Args:
            ids: Tensor of shape (batch_size, seq_len)
        
        Returns:
            List of SELFIES strings
        """
        if len(ids.shape) == 1:
            return [self.decode_one(ids)]
        
        selfies_list = []
        for i in range(ids.shape[0]):
            selfies_list.append(self.decode_one(ids[i]))
        
        return selfies_list
    
    def corrupt_one(self, selfies_str: str) -> torch.Tensor:
        """
        Apply corruption to SELFIES for robust training
        
        Corruption strategies:
        1. Random token deletion
        2. Random token insertion from vocabulary
        3. Token swapping
        
        Args:
            selfies_str: SELFIES string
        
        Returns:
            Corrupted token tensor
        """
        tokens = list(sf.split_selfies(selfies_str))
        total_length = len(tokens) + 2  # +2 for SOS and EOS
        
        if total_length > self.max_len:
            return self.encode_one(selfies_str)
        
        # Corruption parameters
        r = random.random()
        
        # 1. Token deletion (30% chance)
        if r < 0.3 and len(tokens) > 2:
            n_delete = random.randint(1, min(3, len(tokens) - 1))
            delete_positions = random.sample(range(len(tokens)), n_delete)
            tokens = [t for i, t in enumerate(tokens) if i not in delete_positions]
        
        # 2. Token insertion (30% chance)
        elif r < 0.6:
            n_insert = random.randint(1, min(3, self.max_len - total_length))
            for _ in range(n_insert):
                # Insert random token from vocabulary
                random_token_id = random.randint(3, self.vocab_size - 1)
                random_token = self.idtotok[random_token_id]
                insert_pos = random.randint(0, len(tokens))
                tokens.insert(insert_pos, random_token)
        
        # 3. Token swapping (40% chance)
        else:
            if len(tokens) >= 2:
                n_swaps = random.randint(1, min(2, len(tokens) // 2))
                for _ in range(n_swaps):
                    i, j = random.sample(range(len(tokens)), 2)
                    tokens[i], tokens[j] = tokens[j], tokens[i]
        
        # Reconstruct SELFIES and encode
        corrupted_selfies = ''.join(tokens)
        return self.encode_one(corrupted_selfies)
    
    def corrupt(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """
        Apply corruption to SELFIES string(s)
        
        Args:
            inputs: Single SELFIES string or list of SELFIES strings
        
        Returns:
            Tensor of corrupted token IDs
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        tensors = []
        for selfies_str in inputs:
            tensors.append(self.corrupt_one(selfies_str))
        
        return torch.cat(tensors, dim=0)
    
    def __len__(self):
        return self.vocab_size


# Backward compatibility: keep old class name as alias
regexTokenizer = SELFIESTokenizer