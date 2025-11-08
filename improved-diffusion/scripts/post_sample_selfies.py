"""
Post-processing script to repair invalid molecules using correction model
Adapted for SELFIES representations
"""

import argparse
import os
import torch as th
import selfies as sf
from rdkit import Chem
from transformers import set_seed
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from mytokenizers_selfies import SELFIESTokenizer
from mydatasets_selfies import ChEBIdataset


def main():
    set_seed(121)
    args = create_argparser().parse_args()
    
    logger.configure()
    args.sigma_small = True
    
    print("="*60)
    print("SELFIES Molecule Repair Tool")
    print("="*60)
    
    # Initialize tokenizer
    tokenizer = SELFIESTokenizer(
        vocab_path='../../datasets/SELFIES/selfies_vocab.txt',
        max_len=args.ml
    )
    
    # Initialize correction model
    print("\nLoading correction model...")
    model = TransformerNetModel2(
        in_channels=32,
        model_channels=128,
        dropout=0.1,
        use_checkpoint=False,
        config_name='bert-base-uncased',
        training_mode='e2e',
        vocab_size=len(tokenizer),
        experiment_mode='lm',
        logits_mode=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=12,
    )
    
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(0, 400, 20)],
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
    )
    
    print(f"Loading model from: {args.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {pytorch_total_params:,}")
    
    model.to(dist_util.dev())
    model.eval()
    
    # Load invalid molecules to repair
    print("\n" + "="*60)
    print("Loading invalid molecules...")
    print("="*60)
    
    with open('../../tempbadmols.txt') as f:
        content = [l.strip().split('\t') for l in f.readlines()]
        # Convert SMILES to SELFIES for repair
        selfies_list = []
        orders = []
        for order, smiles in content:
            try:
                selfies = sf.encoder(smiles)
                selfies_list.append(selfies)
                orders.append(order)
            except:
                print(f"Warning: Cannot convert SMILES to SELFIES: {smiles}")
                # Use corrupted version as-is
                selfies_list.append(smiles)
                orders.append(order)
    
    args.num_samples = len(selfies_list)
    print(f"Found {args.num_samples} molecules to repair")
    
    # Encode as noise
    noise = model.word_embedding(tokenizer(selfies_list).to(dist_util.dev()))
    print(f"Noise shape: {noise.shape}")
    
    # Repair molecules
    print("\n" + "="*60)
    print("Repairing molecules...")
    print("="*60)
    
    allsample = []
    num_done = 0
    
    while num_done < args.num_samples:
        idend = min(num_done + args.batch_size, args.num_samples)
        print(f'Processing {num_done} : {idend}')
        
        startnoise = noise[num_done:idend]
        desc_state = th.zeros(idend - num_done, 216, 768).to(dist_util.dev())
        desc_mask = th.ones(idend - num_done, 216).to(dist_util.dev())
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample_shape = (idend - num_done, args.ml, model.in_channels)
        
        sample = sample_fn(
            model,
            sample_shape,
            noise=startnoise,
            clip_denoised=args.clip_denoised,
            denoised_fn=None,
            model_kwargs={},
            top_p=args.top_p,
            progress=True,
            desc=(desc_state, desc_mask)
        )
        allsample.append(sample)
        num_done = idend
    
    sample = th.cat(allsample, dim=0)
    
    # Decode to SELFIES
    print("\nDecoding repaired molecules...")
    x_t = th.tensor(sample).cuda()
    logits = model.get_logits(x_t)
    cands = th.topk(logits, k=1, dim=-1)
    sample = cands.indices.squeeze(-1)
    
    repaired_selfies = tokenizer.decode(sample)
    
    # Convert to SMILES and validate
    print("\n" + "="*60)
    print("Validating repaired molecules...")
    print("="*60)
    
    repaired_smiles = []
    valid_count = 0
    
    for selfies_str in repaired_selfies:
        selfies_clean = selfies_str.replace('[PAD]', '').replace('[SOS]', '').replace('[EOS]', '')
        try:
            smiles = sf.decoder(selfies_clean)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
            repaired_smiles.append(smiles)
        except:
            repaired_smiles.append("*")
    
    print(f"✓ Repaired valid molecules: {valid_count}/{len(repaired_smiles)}")
    
    # Save repaired molecules
    with open(args.outputdir, 'w') as f:
        for i, smiles in enumerate(repaired_smiles):
            selfies_clean = repaired_selfies[i].replace('[PAD]', '').replace('[SOS]', '').replace('[EOS]', '')
            f.write(f"{orders[i]}\t{selfies_clean}\t{smiles}\n")
    
    print(f"\n✓ Saved repaired molecules to: {args.outputdir}")
    
    # Merge with original results
    print("\n" + "="*60)
    print("Merging with original results...")
    print("="*60)
    
    # Load original results
    with open('../../selfies_generation_results.txt') as f:
        lines = f.readlines()
        header = lines[0]
        content = [line.strip().split('\t') for line in lines[1:]]
    
    # Load repaired molecules
    with open(args.outputdir) as f:
        repaired = {int(line.split('\t')[0]): line.strip().split('\t') 
                   for line in f.readlines()}
    
    # Replace invalid molecules with repaired ones
    change_count = 0
    for idx, repair_data in repaired.items():
        if idx < len(content):
            _, repaired_selfies, repaired_smiles = repair_data
            # Check if repair was successful
            if repaired_smiles != "*":
                mol = Chem.MolFromSmiles(repaired_smiles)
                if mol is not None:
                    content[idx][0] = repaired_selfies
                    content[idx][1] = repaired_smiles
                    change_count += 1
    
    # Save merged results
    with open('../../selfies_generation_results_repaired.txt', 'w') as f:
        f.write(header)
        for line in content:
            f.write('\t'.join(line) + '\n')
    
    print(f"✓ Repaired {change_count} molecules")
    print(f"✓ Saved to: ../../selfies_generation_results_repaired.txt")
    
    # Final validation
    print("\n" + "="*60)
    print("Final Validation:")
    print("="*60)
    
    final_valid = sum(1 for line in content if line[1] != "*" and 
                     Chem.MolFromSmiles(line[1]) is not None)
    print(f"✓ Total valid molecules: {final_valid}/{len(content)}")
    print(f"✓ Validity: {100*final_valid/len(content):.2f}%")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen"
    )
    
    text_defaults = dict(
        modality='text',
        dataset_name='wikitext',
        dataset_config_name='wikitext-2-raw-v1',
        experiment='gpt2_pre_compress',
        model_arch='trans-unet',
        preprocessing_num_workers=1,
        emb_scale_factor=1.0,
        clamp='clamp',
        split='test',
        model_path='../../correction_checkpoints_selfies/PLAIN_ema_0.9999_200000.pt',
        use_ddim=False,
        ml=384,
        batch_size=64,
        num_samples=3300,
        top_p=1.0,
        out_dir='generation_outputs',
        outputdir='../../tempregeneratebad.txt'
    )
    
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
