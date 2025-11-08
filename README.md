# Text-Guided Diffusion Model for SELFIES Molecule Generation

This project implements a **text-guided diffusion model** for generating novel molecules, adapted from the original [tgm-dlm](https://github.com/Deno-V/tgm-dlm.git) repository.  
It has been **significantly modified** to operate using the **SELFIES (SELF-referencIng Embedded Strings)** molecular representation instead of SMILES.

---

## Key Advantages

- **98+% syntactic validity** of generated molecules (by design — SELFIES guarantees valid molecular strings)
- **SciBERT-based text guidance:** Generates molecules aligned with textual property descriptions
- **Fully automated SELFIES data pipeline**
- **Comprehensive evaluation metrics** for molecule quality and diversity

---

## Setup and Installation

### 1️-- Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
2️----Install Dependencies
Make sure you have requirements.txt in the root directory. Then run:

```
```bash
Copy code
pip install -r requirements.txt
pip install selfies matplotlib rdkit-pypi
```
Note: Ensure that your PyTorch version matches your hardware (CPU or CUDA).

Data Preparation Workflow
This workflow converts the original SMILES dataset into SELFIES, builds a vocabulary, and generates SciBERT text embeddings.

Prerequisite
Place your raw data files inside:

```bash
swift
Copy code
datasets/SMILES/
```
Each file should be a .tsv file with CID, SMILES, and description columns:
```bash
Copy code
train.txt
validation.txt
test.txt
```

Step 2.1 — Convert SMILES → SELFIES & Build Vocabulary
```bash
Copy code
cd improved-diffusion/scripts/
python convert_smiles_to_selfies.py
```
Output:

```bash
swift
Copy code
datasets/SELFIES/
├── train.txt
├── test.txt
├── validation.txt
└── selfies_vocab.txt
```
Step 2.2 — Clean the SELFIES Dataset
Validate and clean malformed SELFIES strings.

```bash
Copy code
python clean_selfies_dataset.py
```
Output: Cleaned .txt files in datasets/SELFIES/.

Step 2.3 — Generate Text Embeddings (SciBERT)
Use SciBERT (located at ../../scibert/) to create text embeddings.

```bash
Copy code
# Training split
python process_text_selfies.py -i train_val_256

# Validation split
python process_text_selfies.py -i validation_256

# Test split
python process_text_selfies.py -i test
Output: .pt files in datasets/SELFIES/
(e.g., train_val_256_scibert_desc.pt)
```

Training the Model
To start training the SELFIES-based diffusion model:

```bash
Copy code
cd improved-diffusion/scripts/
python train_selfies.py
Checkpoints and logs are automatically saved to:

Copy code
checkpoints/
Optional: Training with Corruption/Masking
bash
Copy code
python train_correct_withmask_selfies.py
Useful for post-sample repair.
```
Generating Molecules (Sampling)
Generate new molecules from text descriptions using a trained checkpoint:

```bash
Copy code
python text_sample_selfies.py \
--model_path ../../checkpoints/<your_model_checkpoint.pt> \
--num_samples 10000 \
--output_file ../../selfies_generation_results.txt
```
Arguments:
```bash
--model_path: Path to model checkpoint (e.g., ema_0.9999_200000.pt)

--num_samples: Number of molecules to generate (10k–30k recommended)

--output_file: Output file to save generated SELFIES & SMILES
```
Evaluating Generation Results
Run the evaluation script:

```bash
Copy code
python evaluate_selfies_generation.py \
--generated ../../selfies_generation_results.txt \
--ground_truth ../../datasets/SELFIES/test.txt \
--output_dir ./evaluation_results
```
Outputs:

Console: Detailed metrics report

evaluation_results/evaluation_metrics.json: Metrics summary (Validity, Uniqueness, Novelty, Lipinski Ro5, etc.)

evaluation_results/plots/: Publication-quality plots:

property_distributions.png

metrics_summary.png

(Optional) Post-Processing / Repair
If you trained a correction model, you can repair invalid SELFIES using:

```bash
Copy code
python post_sample_selfies.py \
--model_path ../../correction_checkpoints/<your_correction_model.pt> \
--input_file ../../selfies_generation_results_invalid.txt \
--output_file ../../selfies_repaired_results.txt
Key Project Files (SELFIES Migration)
Core Logic
```
mytokenizers_selfies.py – SELFIES tokenizer class (SELFIESTokenizer)

mydatasets_selfies.py – Custom ChEBI dataset loader

Data Pipeline

convert_smiles_to_selfies.py – SMILES → SELFIES conversion

clean_selfies_dataset.py – SELFIES validation and cleanup

process_text_selfies.py – SciBERT embedding generation

Model Execution

train_selfies.py – Main training script

train_correct_withmask_selfies.py – Masked training variant

text_sample_selfies.py – Molecule sampling

evaluate_selfies_generation.py – Evaluation metrics

post_sample_selfies.py – Repair invalid samples

Acknowledgements
This work is a modification of the original Text-Guided Mask Denoising Language Model (TGM-DLM).

The primary modification adapts the model from SMILES to SELFIES, ensuring 100% valid molecular generation and improved robustness for text-guided molecule design.

