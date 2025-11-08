"""
Generate SELFIES samples from a trained diffusion model.
Modified from original SMILES sampling script.
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
    print("Creating model and diffusion...")
    print("="*60)
    
    # Initialize SELFIES tokenizer
    tokenizer = SELFIESTokenizer(
        vocab_path='../../datasets/SELFIES/selfies_vocab.txt',
        max_len=256
    )
    
    print(f'SELFIES Vocabulary size: {len(tokenizer)}')
    
    # Initialize model
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
    
    # Initialize diffusion
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(0, 2000, 10)],
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
    print(f'Model parameter count: {pytorch_total_params:,}')

    model.to(dist_util.dev())
    model.eval()

    print("="*60)
    print(f'Loading {args.split} dataset...')
    print("="*60)

    # Load dataset
    test_dataset = ChEBIdataset(
        dir='../../datasets/SELFIES/',
        selfies_tokenizer=tokenizer,
        split=args.split,
        replace_desc=False
    )
    
    print(f'Dataset size: {len(test_dataset)}')
    print(f'Description state shape: {test_dataset[0]["desc_state"].shape}')
    
    # Prepare data
    desc_data = [
        (
            test_dataset[i]['desc_state'],
            test_dataset[i]['desc_mask'],
            test_dataset[i]['selfies']  # Changed from 'smiles'
        )
        for i in range(args.num_samples)
    ]
    ground_truth = [d[2] for d in desc_data]

    print("="*60)
    print("Generating samples...")
    print("="*60)
    
    # Sampling
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    
    all_samples = []
    num_done = 0
    
    while num_done < args.num_samples:
        id_end = min(num_done + args.batch_size, args.num_samples)
        print(f'Generating samples {num_done} to {id_end}...')
        
        desc_state = th.cat([d[0] for d in desc_data[num_done:id_end]], dim=0)
        desc_mask = th.cat([d[1] for d in desc_data[num_done:id_end]], dim=0)
        
        sample_shape = (id_end - num_done, 256, model.in_channels)
        
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=None,
            model_kwargs={},
            top_p=args.top_p,
            progress=True,
            desc=(desc_state, desc_mask)
        )
        
        all_samples.append(sample)
        num_done = id_end
    
    # Decode samples
    sample = th.cat(all_samples, dim=0)
    print(f'Decoding {sample.shape[0]} samples...')
    
    x_t = th.tensor(sample).cuda()
    logits = model.get_logits(x_t)
    cands = th.topk(logits, k=1, dim=-1)
    sample_ids = cands.indices.squeeze(-1)
    
    # Decode to SELFIES
    selfies_list = tokenizer.decode(sample_ids)
    
    # Convert SELFIES to SMILES for validation
    print("="*60)
    print("Converting SELFIES to SMILES...")
    print("="*60)
    
    smiles_list = []
    for selfies_str in selfies_list:
        # Clean SELFIES string
        selfies_clean = selfies_str.replace('[PAD]', '').replace('[SOS]', '').replace('[EOS]', '')
        try:
            smiles = sf.decoder(selfies_clean)
            smiles_list.append(smiles)
        except Exception as e:
            print(f"Warning: Failed to decode SELFIES: {selfies_clean[:50]}...")
            smiles_list.append("*")
    
    # Save results
    print("="*60)
    print(f"Saving results to: {args.outputdir}")
    print("="*60)
    
    with open(args.outputdir, 'w') as f:
        f.write("Generated_SELFIES\tGenerated_SMILES\tGround_Truth_SELFIES\n")
        for i, (selfies_str, smiles, gt_selfies) in enumerate(zip(selfies_list, smiles_list, ground_truth)):
            if i == 0:
                print("Sample output:")
                print(f"  SELFIES: {selfies_str[:100]}...")
                print(f"  SMILES:  {smiles}")
                print(f"  Ground truth: {gt_selfies[:100]}...")
            
            selfies_clean = selfies_str.replace('[PAD]', '').replace('[SOS]', '').replace('[EOS]', '')
            f.write(f"{selfies_clean}\t{smiles}\t{gt_selfies}\n")
    
    # Validate generated molecules
    print("="*60)
    print("Validating generated molecules...")
    print("="*60)
    
    valid_count = 0
    invalid_molecules = []
    
    for i, smiles in enumerate(smiles_list):
        if smiles == "*":
            invalid_molecules.append((i, "SELFIES decode failed", ""))
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_molecules.append((i, smiles, "Invalid SMILES"))
        else:
            valid_count += 1
    
    validity = 100.0 * valid_count / len(smiles_list)
    
    print(f"\n✓ Valid molecules: {valid_count}/{len(smiles_list)} ({validity:.2f}%)")
    print(f"✗ Invalid molecules: {len(invalid_molecules)}")
    
    # Save invalid molecules
    if invalid_molecules:
        invalid_path = args.outputdir.replace('.txt', '_invalid.txt')
        with open(invalid_path, 'w') as f:
            f.write("Index\tGenerated_SMILES\tError\n")
            for idx, smiles, error in invalid_molecules:
                f.write(f"{idx}\t{smiles}\t{error}\n")
        print(f"Invalid molecules saved to: {invalid_path}")
    
    print("="*60)
    print("✓ Generation complete!")
    print("="*60)


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
        model_path='../../checkpoints_selfies/PLAIN_ema_0.9999_200000.pt',
        use_ddim=False,
        batch_size=64,
        num_samples=100,
        top_p=1.0,
        out_dir='generation_outputs_selfies',
        outputdir='../../selfies_generation_results.txt'
    )
    
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
