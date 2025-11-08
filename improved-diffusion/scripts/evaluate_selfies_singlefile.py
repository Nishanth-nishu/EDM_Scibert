#!/usr/bin/env python3
"""
SELFIES Molecular Generation Evaluator (Single-File Version) — Updated
- Novelty is computed vs training dataset (provided via --train_file)
- Adds research-style plots and additional metrics:
    * Validity, Uniqueness, Novelty (vs training)
    * Molecular property distributions (MW, LogP, TPSA, QED, rotatable bonds)
    * Scatter plots (MW vs LogP, QED vs MW)
    * Internal diversity (sampled pairwise Tanimoto distribution)
    * Nearest-neighbour similarity to training set (max Tanimoto per generated mol)
    * Scaffold novelty (Bemis-Murcko scaffold overlap)
- Saves JSON metrics and PNG plots
"""

import argparse
import os
import json
import warnings
from collections import Counter
from math import comb
import random
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit import RDLogger

# suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


def decode_selfies_safe(s):
    try:
        if s and s != '*' and isinstance(s, str):
            return sf.decoder(s)
    except Exception:
        return None
    return None


def canonical_smiles_from_smiles(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
    return None


def get_bemis_scaffold(mol):
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


class MolecularEvaluatorSingleFile:
    def __init__(self, generated_file, train_file, output_dir='./evaluation_results',
                 max_internal_pairs=5000, max_train_sample=10000, random_seed=42):
        """
        :param generated_file: file with Generated_SELFIES, Generated_SMILES, Ground_Truth_SELFIES
        :param train_file: training dataset file (SELFIES or SMILES). REQUIRED for correct novelty.
        :param output_dir: where to save JSON and plots
        :param max_internal_pairs: sample size for internal diversity pair computations
        :param max_train_sample: sample size limit for training set when computing nearest-neighbour similarities
        :param random_seed: seed for reproducible sampling in pairwise computations
        """
        if train_file is None:
            raise ValueError("train_file is required to compute novelty vs training set (use --train_file).")
        self.generated_file = generated_file
        self.train_file = train_file
        self.output_dir = output_dir
        self.max_internal_pairs = max_internal_pairs
        self.max_train_sample = max_train_sample
        self.random_seed = random_seed
        os.makedirs(self.output_dir, exist_ok=True)

        # load data
        self.generated_data = self.load_generated_data()
        self.training_smiles = self.load_training_smiles(self.train_file)

    # -------------------------
    # Loading functions
    # -------------------------
    def load_generated_data(self):
        print("Loading generated molecules...")
        # try tab, then whitespace fallback
        df = pd.read_csv(self.generated_file, sep='\t', engine='python', on_bad_lines='skip')
        if df.shape[1] == 1:
            df = pd.read_csv(self.generated_file, sep='\s+', engine='python', on_bad_lines='skip')

        expected_cols = ['Generated_SELFIES', 'Generated_SMILES', 'Ground_Truth_SELFIES']
        if not all(col in df.columns for col in expected_cols):
            print(f"⚠ Columns found: {df.columns.tolist()}")
            raise ValueError(f"File must contain columns: {expected_cols}")

        print(f"✓ Loaded {len(df)} generated molecules from {self.generated_file}")
        return df

    def load_training_smiles(self, train_file):
        """
        Load training molecules and canonicalize to SMILES.
        Accepts:
         - tab-separated file with column 'SELFIES' or 'SMILES'
         - single-column file: auto-detect SELFIES (contains '[') vs SMILES
        Returns: set of canonical SMILES strings
        """
        print("\nLoading training set and canonicalizing...")

        # attempt to read with pandas
        try:
            df = pd.read_csv(train_file, sep='\t', engine='python', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(train_file, sep='\s+', header=None, engine='python', on_bad_lines='skip')

        # determine column
        if isinstance(df, pd.DataFrame):
            cols = df.columns.tolist()
            col = None
            if 'SELFIES' in cols:
                col = 'SELFIES'
            elif 'SMILES' in cols:
                col = 'SMILES'
            elif df.shape[1] == 1:
                col = df.columns[0]
            else:
                # try to find likely SELFIES column
                for c in cols:
                    sample_vals = df[c].dropna().astype(str).head(20).tolist()
                    if any('[' in v for v in sample_vals):
                        col = c
                        break
                if col is None:
                    col = cols[0]
        else:
            raise ValueError("Could not read training file into DataFrame.")

        smiles_set = set()
        for val in df[col].astype(str).dropna().tolist():
            s = val.strip()
            if s == '':
                continue
            try:
                # heuristic: presence of '[' likely SELFIES
                if '[' in s:
                    decoded = decode_selfies_safe(s)
                    if decoded:
                        s_smiles = decoded
                    else:
                        continue
                else:
                    s_smiles = s
                can = canonical_smiles_from_smiles(s_smiles)
                if can:
                    smiles_set.add(can)
            except Exception:
                continue

        print(f"✓ Loaded {len(smiles_set)} canonical training molecules from {train_file}")
        return smiles_set

    # -------------------------
    # Validity
    # -------------------------
    def calculate_validity(self):
        print("\n" + "=" * 60)
        print("1. VALIDITY ANALYSIS")
        print("=" * 60)

        valid_mols = []
        invalid_indices = []

        for idx, row in self.generated_data.iterrows():
            selfies = row['Generated_SELFIES']
            # prefer provided Generated_SMILES if valid, else decode selfies
            smiles_candidate = row.get('Generated_SMILES', None)
            if isinstance(smiles_candidate, str) and smiles_candidate.strip() != '':
                mol = Chem.MolFromSmiles(smiles_candidate)
                if mol:
                    valid_mols.append((idx, Chem.MolToSmiles(mol, canonical=True), mol))
                    continue
            # fallback to decode SELFIES
            decoded = decode_selfies_safe(selfies)
            mol = Chem.MolFromSmiles(decoded) if decoded else None
            if mol:
                valid_mols.append((idx, Chem.MolToSmiles(mol, canonical=True), mol))
            else:
                invalid_indices.append(idx)

        n_total = len(self.generated_data)
        n_valid = len(valid_mols)
        validity = 100.0 * n_valid / n_total if n_total > 0 else 0.0
        print(f"✓ Valid molecules: {n_valid}/{n_total} ({validity:.2f}%)")
        return {'validity': validity, 'n_valid': n_valid, 'n_total': n_total, 'valid_mols': valid_mols}

    # -------------------------
    # Uniqueness
    # -------------------------
    def calculate_uniqueness(self, valid_mols):
        print("\n" + "=" * 60)
        print("2. UNIQUENESS ANALYSIS")
        print("=" * 60)
        smiles_list = [smiles for _, smiles, _ in valid_mols]
        unique_smiles = list(dict.fromkeys(smiles_list))  # preserves order, removes duplicates
        uniqueness = 100.0 * len(unique_smiles) / len(smiles_list) if len(smiles_list) > 0 else 0.0
        print(f"✓ Unique molecules: {len(unique_smiles)}/{len(smiles_list)} ({uniqueness:.2f}%)")
        return {'uniqueness': uniqueness, 'unique_smiles': unique_smiles}

    # -------------------------
    # Novelty vs training
    # -------------------------
    def calculate_novelty_vs_training(self, unique_smiles):
        print("\n" + "=" * 60)
        print("3. NOVELTY ANALYSIS (vs training set)")
        print("=" * 60)

        reference = self.training_smiles
        ref_size = len(reference)
        print(f"Using training set reference size = {ref_size} canonical molecules.")

        novel = []
        valid_count = 0
        for s in unique_smiles:
            mol = Chem.MolFromSmiles(s)
            if not mol:
                continue
            can = Chem.MolToSmiles(mol, canonical=True)
            valid_count += 1
            if can not in reference:
                novel.append(can)

        novelty = 100.0 * len(novel) / valid_count if valid_count > 0 else 0.0
        print(f"✓ Novel molecules (vs training): {len(novel)}/{valid_count} ({novelty:.2f}%)")
        return {'novelty': novelty, 'novel_mols': novel, 'n_reference': ref_size, 'n_valid_unique': valid_count}

    # -------------------------
    # Molecular properties
    # -------------------------
    def calculate_molecular_properties(self, valid_mols):
        print("\n" + "=" * 60)
        print("4. MOLECULAR PROPERTIES")
        print("=" * 60)

        props = {
            'molecular_weight': [], 'logp': [], 'num_hba': [],
            'num_hbd': [], 'num_rotatable_bonds': [], 'tpsa': [],
            'num_aromatic_rings': [], 'qed': []
        }

        for _, _, mol in valid_mols:
            try:
                props['molecular_weight'].append(Descriptors.MolWt(mol))
                props['logp'].append(Crippen.MolLogP(mol))
                props['num_hba'].append(Lipinski.NumHAcceptors(mol))
                props['num_hbd'].append(Lipinski.NumHDonors(mol))
                props['num_rotatable_bonds'].append(Lipinski.NumRotatableBonds(mol))
                props['tpsa'].append(Descriptors.TPSA(mol))
                props['num_aromatic_rings'].append(Descriptors.NumAromaticRings(mol))
                props['qed'].append(QED.qed(mol))
            except Exception:
                continue

        stats = {}
        for k, v in props.items():
            if len(v) > 0:
                stats[k] = {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'count': len(v)}
            else:
                stats[k] = {'mean': None, 'std': None, 'count': 0}
            print(f"{k:20s}: {stats[k]['mean']:.2f} ± {stats[k]['std']:.2f}" if stats[k]['count'] > 0 else f"{k:20s}: n/a")

        return {'properties': props, 'stats': stats}

    # -------------------------
    # Reconstruction accuracy
    # -------------------------
    def calculate_reconstruction_accuracy(self):
        print("\n" + "=" * 60)
        print("5. RECONSTRUCTION ACCURACY")
        print("=" * 60)

        exact = 0
        similar = 0
        total = len(self.generated_data)

        for _, row in self.generated_data.iterrows():
            gen_selfies = row['Generated_SELFIES']
            gt_selfies = row['Ground_Truth_SELFIES']
            gen_smiles = None
            if isinstance(row.get('Generated_SMILES', None), str) and row['Generated_SMILES'].strip() != '':
                gen_smiles = row['Generated_SMILES']
            else:
                gen_smiles = decode_selfies_safe(gen_selfies)
            gt_smiles = decode_selfies_safe(gt_selfies)

            if not gen_smiles or not gt_smiles:
                continue

            mol1 = Chem.MolFromSmiles(gen_smiles)
            mol2 = Chem.MolFromSmiles(gt_smiles)
            if not mol1 or not mol2:
                continue

            # canonical exact match
            try:
                if Chem.MolToSmiles(mol1, canonical=True) == Chem.MolToSmiles(mol2, canonical=True):
                    exact += 1
            except Exception:
                pass

            try:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                if sim > 0.85:
                    similar += 1
            except Exception:
                continue

        print(f"✓ Exact matches: {exact}/{total} ({100*exact/total:.2f}%)")
        print(f"✓ High similarity (>0.85): {similar}/{total} ({100*similar/total:.2f}%)")
        return {'exact': exact, 'similar': similar, 'total': total}

    # -------------------------
    # Fingerprints & similarity helpers
    # -------------------------
    def smiles_to_fp(self, smiles, radius=2, nbits=2048):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

    def compute_training_fps(self, max_sample=None):
        """
        Compute fingerprints for training set. If training set is huge, sample up to max_sample.
        """
        t_smiles = list(self.training_smiles)
        if max_sample and len(t_smiles) > max_sample:
            random.seed(self.random_seed)
            t_smiles = random.sample(t_smiles, max_sample)
        fps = []
        for s in t_smiles:
            fp = self.smiles_to_fp(s)
            if fp:
                fps.append(fp)
        return fps

    def compute_generated_fps(self, unique_smiles):
        fps = []
        smiles_list = []
        for s in unique_smiles:
            fp = self.smiles_to_fp(s)
            if fp:
                fps.append(fp)
                smiles_list.append(s)
        return smiles_list, fps

    def nearest_neighbor_similarity(self, unique_smiles):
        """
        For each generated molecule, compute max Tanimoto similarity to the training set.
        Returns list of max similarities (same order as unique_smiles filtered to valid).
        """
        print("\nComputing nearest-neighbour similarities to training set...")
        train_fps = self.compute_training_fps(max_sample=self.max_train_sample)
        if len(train_fps) == 0:
            print("Warning: no training fingerprints available for NN similarity.")
            return []

        gen_smiles_filtered, gen_fps = self.compute_generated_fps(unique_smiles)
        max_sims = []
        for i, gfp in enumerate(gen_fps):
            max_sim = 0.0
            # iterate over training fps
            for tfp in train_fps:
                sim = DataStructs.TanimotoSimilarity(gfp, tfp)
                if sim > max_sim:
                    max_sim = sim
                    if max_sim == 1.0:
                        break
            max_sims.append(max_sim)
        return {'gen_smiles': gen_smiles_filtered, 'max_sims': max_sims}

    def internal_diversity_sampled(self, unique_smiles):
        """
        Estimate internal diversity by sampling pairwise Tanimoto similarities among generated unique molecules.
        Returns list of sampled similarities.
        """
        print("\nEstimating internal diversity (sampled pairwise Tanimoto)...")
        gen_smiles_filtered, gen_fps = self.compute_generated_fps(unique_smiles)
        n = len(gen_fps)
        if n < 2:
            return []

        # total possible pairs
        total_pairs = n * (n - 1) // 2
        sample_pairs = min(self.max_internal_pairs, total_pairs)
        # If total_pairs is small enough, compute all
        if sample_pairs == total_pairs:
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    sims.append(DataStructs.TanimotoSimilarity(gen_fps[i], gen_fps[j]))
            return sims

        # Otherwise sample random pairs
        random.seed(self.random_seed)
        sims = []
        for _ in range(sample_pairs):
            i, j = random.sample(range(n), 2)
            sims.append(DataStructs.TanimotoSimilarity(gen_fps[i], gen_fps[j]))
        return sims

    # -------------------------
    # Scaffold novelty
    # -------------------------
    def scaffold_novelty(self, unique_smiles):
        """
        Compute Bemis-Murcko scaffolds for training and generated sets and report scaffold novelty:
        fraction of generated scaffolds not present in training scaffolds.
        """
        print("\nComputing scaffold novelty (Bemis-Murcko)...")
        train_scaffolds = set()
        for s in self.training_smiles:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    sc = get_bemis_scaffold(mol)
                    if sc:
                        train_scaffolds.add(sc)
            except Exception:
                continue

        gen_scaffolds = set()
        for s in unique_smiles:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    sc = get_bemis_scaffold(mol)
                    if sc:
                        gen_scaffolds.add(sc)
            except Exception:
                continue

        if len(gen_scaffolds) == 0:
            novelty = 0.0
        else:
            novel_scaffolds = [sc for sc in gen_scaffolds if sc not in train_scaffolds]
            novelty = 100.0 * len(novel_scaffolds) / len(gen_scaffolds)
        print(f"✓ Generated scaffolds: {len(gen_scaffolds)}, novel scaffolds: {len([sc for sc in gen_scaffolds if sc not in train_scaffolds])} ({novelty:.2f}%)")
        return {'n_gen_scaffolds': len(gen_scaffolds), 'n_train_scaffolds': len(train_scaffolds),
                'novel_scaffold_pct': novelty, 'novel_scaffolds': list(sc for sc in gen_scaffolds if sc not in train_scaffolds)}

    # -------------------------
    # Plotting
    # -------------------------
    def plot_summary_graphs(self, results):
        print("\nGenerating plots...")
        out = self.output_dir
        # overall metrics bar
        metrics = {
            "Validity (%)": results['validity']['validity'],
            "Uniqueness (%)": results['uniqueness']['uniqueness'],
            "Novelty (%)": results['novelty']['novelty']
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(metrics.keys(), metrics.values())
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percentage")
        ax.set_title("Overall Metrics")
        for i, v in enumerate(metrics.values()):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center")
        plt.tight_layout()
        fig_path = os.path.join(out, "summary_metrics.png")
        fig.savefig(fig_path)
        plt.close(fig)

        # properties histograms
        props = results['properties']['properties']
        prop_names_to_plot = ['molecular_weight', 'logp', 'tpsa', 'qed', 'num_rotatable_bonds']
        for name in prop_names_to_plot:
            vals = props.get(name, [])
            if len(vals) == 0:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(vals, bins=40, edgecolor='black')
            ax.set_title(f"{name} distribution")
            ax.set_xlabel(name)
            ax.set_ylabel("Count")
            plt.tight_layout()
            fig.savefig(os.path.join(out, f"hist_{name}.png"))
            plt.close(fig)

        # scatter MW vs LogP
        mw = props.get('molecular_weight', [])
        logp = props.get('logp', [])
        if len(mw) == len(logp) and len(mw) > 1:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(mw, logp, s=8)
            ax.set_xlabel("Molecular Weight")
            ax.set_ylabel("LogP")
            ax.set_title("MW vs LogP")
            plt.tight_layout()
            fig.savefig(os.path.join(out, "scatter_mw_logp.png"))
            plt.close(fig)

        # scatter QED vs MW
        qed = props.get('qed', [])
        if len(qed) == len(mw) and len(qed) > 1:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(mw, qed, s=8)
            ax.set_xlabel("Molecular Weight")
            ax.set_ylabel("QED")
            ax.set_title("QED vs MW")
            plt.tight_layout()
            fig.savefig(os.path.join(out, "scatter_qed_mw.png"))
            plt.close(fig)

        # internal diversity histogram
        int_div = results.get('internal_diversity', [])
        if len(int_div) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(int_div, bins=40, edgecolor='black')
            ax.set_title("Internal Diversity (sampled pairwise Tanimoto)")
            ax.set_xlabel("Tanimoto similarity")
            ax.set_ylabel("Count")
            plt.tight_layout()
            fig.savefig(os.path.join(out, "hist_internal_diversity.png"))
            plt.close(fig)

        # nearest neighbour similarity histogram
        nn = results.get('nearest_neighbour', {}).get('max_sims', [])
        if len(nn) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(nn, bins=40, edgecolor='black')
            ax.set_title("Nearest-neighbour similarity to training set (max Tanimoto)")
            ax.set_xlabel("Max Tanimoto similarity")
            ax.set_ylabel("Count")
            plt.tight_layout()
            fig.savefig(os.path.join(out, "hist_nn_similarity.png"))
            plt.close(fig)

        # scaffold novelty bar
        sc = results.get('scaffold_novelty', {})
        if sc:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ['Train scaffolds', 'Gen scaffolds', 'Novel gen scaffolds']
            train_n = sc.get('n_train_scaffolds', 0)
            gen_n = sc.get('n_gen_scaffolds', 0)
            novel_pct = sc.get('novel_scaffold_pct', 0.0)
            novel_n = int(round(gen_n * novel_pct / 100.0)) if gen_n > 0 else 0
            vals = [train_n, gen_n, novel_n]
            ax.bar(labels, vals)
            ax.set_title("Scaffold counts and novel scaffold count")
            plt.tight_layout()
            fig.savefig(os.path.join(out, "bar_scaffold_novelty.png"))
            plt.close(fig)

        print(f"✓ Plots saved to {self.output_dir}")

    # -------------------------
    # Save JSON
    # -------------------------
    # -------------------------------------------------------------------------
    # JSON-safe saving (robust version)
    # -------------------------------------------------------------------------
    def save_json_safe(self, results, out_path):
        """Save results to JSON, converting RDKit Mol and numpy objects safely."""

        def convert(obj):
            # 1. Handle RDKit molecules anywhere
            try:
                from rdkit.Chem.rdchem import Mol
                if isinstance(obj, Mol):
                    return Chem.MolToSmiles(obj, canonical=True)
            except Exception:
                pass

            # 2. Handle common numeric types
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()

            # 3. Containers: sets, dicts, lists, tuples
            elif isinstance(obj, (set, Counter)):
                return [convert(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]

            # 4. Anything else: return as-is or string
            try:
                json.dumps(obj)  # test if JSON-safe
                return obj
            except Exception:
                return str(obj)

        # 5. Write the fully converted structure
        converted = convert(results)
        with open(out_path, 'w') as f:
            json.dump(converted, f, indent=2)
        print(f"✓ Results saved to {out_path}")


    # -------------------------
    # Run evaluation
    # -------------------------
    def run_evaluation(self):
        start_time = time.time()
        results = {}

        # 1. Validity
        validity = self.calculate_validity()
        results['validity'] = validity
        if validity['n_valid'] == 0:
            print("❌ No valid molecules found. Exiting.")
            return results

        # 2. Uniqueness
        uniq = self.calculate_uniqueness(validity['valid_mols'])
        results['uniqueness'] = uniq

        # 3. Novelty (vs training)
        novelty = self.calculate_novelty_vs_training(uniq['unique_smiles'])
        results['novelty'] = novelty

        # 4. Properties
        props = self.calculate_molecular_properties(validity['valid_mols'])
        results['properties'] = props

        # 5. Reconstruction
        recon = self.calculate_reconstruction_accuracy()
        results['reconstruction'] = recon

        # 6. Internal diversity (sampled)
        internal_div = self.internal_diversity_sampled(uniq['unique_smiles'])
        results['internal_diversity'] = internal_div
        if len(internal_div) > 0:
            results['internal_diversity_summary'] = {
                'mean': float(np.mean(internal_div)), 'std': float(np.std(internal_div)), 'count': len(internal_div)
            }

        # 7. Nearest neighbour similarity to training
        nn = self.nearest_neighbor_similarity(uniq['unique_smiles'])
        results['nearest_neighbour'] = nn
        if 'max_sims' in nn and len(nn['max_sims']) > 0:
            results['nn_summary'] = {'mean': float(np.mean(nn['max_sims'])), 'std': float(np.std(nn['max_sims'])),
                                     'count': len(nn['max_sims'])}

        # 8. Scaffold novelty
        sc = self.scaffold_novelty(uniq['unique_smiles'])
        results['scaffold_novelty'] = sc

        # Save JSON results
        out_json = os.path.join(self.output_dir, 'evaluation_metrics_with_plots.json')
        self.save_json_safe(results, out_json)

        # Generate plots
        try:
            self.plot_summary_graphs(results)
        except Exception as e:
            print(f"⚠ Could not create plots: {e}")

        elapsed = time.time() - start_time
        print(f"\nEvaluation complete in {elapsed/60.0:.2f} minutes.")
        return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SELFIES generation (single-file) with training-based novelty and plots")
    parser.add_argument('--generated', required=True,
                        help="Path to the generated results file (contains Generated_SELFIES, Generated_SMILES, Ground_Truth_SELFIES)")
    parser.add_argument('--train_file', required=True,
                        help="Path to training dataset file (SELFIES or SMILES). Novelty will be computed vs this file.")
    parser.add_argument('--output_dir', default='./evaluation_results', help="Directory to save results and plots")
    parser.add_argument('--max_internal_pairs', type=int, default=5000,
                        help="Max sampled pairs for internal diversity (keeps computation bounded).")
    parser.add_argument('--max_train_sample', type=int, default=10000,
                        help="Max number of training molecules to sample for nearest-neighbour similarity (for speed).")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for sampling in similarity/diversity estimates")
    args = parser.parse_args()

    evaluator = MolecularEvaluatorSingleFile(
        generated_file=args.generated,
        train_file=args.train_file,
        output_dir=args.output_dir,
        max_internal_pairs=args.max_internal_pairs,
        max_train_sample=args.max_train_sample,
        random_seed=args.random_seed
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
