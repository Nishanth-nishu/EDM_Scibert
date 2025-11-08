"""
Quick evaluation script for SELFIES generation results
Provides essential metrics without heavy computation
"""

import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from collections import Counter
import selfies as sf


def quick_evaluate(generated_file):
    """Fast evaluation with essential metrics"""
    
    print("="*60)
    print("QUICK EVALUATION")
    print("="*60)
    
    # Load data
    print("\nLoading generated molecules...")
    df = pd.read_csv(generated_file, sep='\t')
    print(f"✓ Loaded {len(df)} molecules\n")
    
    # Metrics
    valid_smiles = []
    invalid_count = 0
    
    properties = {
        'mw': [],
        'logp': [],
        'hba': [],
        'hbd': [],
        'qed': []
    }
    
    ro5_compliant = 0
    
    print("Computing metrics...")
    for idx, row in df.iterrows():
        smiles = row.get('Generated_SMILES', row.iloc[1])
        
        if pd.isna(smiles) or smiles == '*':
            invalid_count += 1
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_count += 1
            continue
        
        valid_smiles.append(smiles)
        
        # Properties
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        
        properties['mw'].append(mw)
        properties['logp'].append(logp)
        properties['hba'].append(hba)
        properties['hbd'].append(hbd)
        
        # Lipinski Ro5
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hba > 10: violations += 1
        if hbd > 5: violations += 1
        
        if violations == 0:
            ro5_compliant += 1
    
    # Calculate metrics
    n_total = len(df)
    n_valid = len(valid_smiles)
    n_unique = len(set(valid_smiles))
    
    validity = 100.0 * n_valid / n_total
    uniqueness = 100.0 * n_unique / n_valid if n_valid > 0 else 0
    ro5_compliance = 100.0 * ro5_compliant / n_valid if n_valid > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n1. VALIDITY:")
    print(f"   Valid molecules: {n_valid}/{n_total} ({validity:.2f}%)")
    
    print(f"\n2. UNIQUENESS:")
    print(f"   Unique molecules: {n_unique}/{n_valid} ({uniqueness:.2f}%)")
    
    print(f"\n3. LIPINSKI Ro5 COMPLIANCE:")
    print(f"   Compliant molecules: {ro5_compliant}/{n_valid} ({ro5_compliance:.2f}%)")
    
    print(f"\n4. MOLECULAR PROPERTIES (Mean):")
    print(f"   Molecular Weight: {sum(properties['mw'])/len(properties['mw']):.2f}")
    print(f"   LogP: {sum(properties['logp'])/len(properties['logp']):.2f}")
    print(f"   H-Bond Acceptors: {sum(properties['hba'])/len(properties['hba']):.2f}")
    print(f"   H-Bond Donors: {sum(properties['hbd'])/len(properties['hbd']):.2f}")
    
    print("\n" + "="*60)
    
    # Save summary
    summary_file = generated_file.replace('.txt', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Validity: {validity:.2f}%\n")
        f.write(f"Uniqueness: {uniqueness:.2f}%\n")
        f.write(f"Ro5 Compliance: {ro5_compliance:.2f}%\n")
        f.write(f"Avg MW: {sum(properties['mw'])/len(properties['mw']):.2f}\n")
        f.write(f"Avg LogP: {sum(properties['logp'])/len(properties['logp']):.2f}\n")
    
    print(f"✓ Summary saved to: {summary_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Generated results file')
    args = parser.parse_args()
    
    quick_evaluate(args.file)
