# Text-Guided Diffusion Model for SELFIES Molecule Generation

This project implements a **text-guided diffusion model** for generating novel molecules, adapted from the original [tgm-dlm](https://github.com/Deno-V/tgm-dlm.git) repository.
It has been **significantly modified** to operate using the **SELFIES (SELF-referencIng Embedded Strings)** molecular representation instead of SMILES.

---

## Key Advantages

* **100% syntactic validity** of generated molecules (SELFIES guarantees valid molecular strings)
* **SciBERT-based text guidance** for molecule-property alignment
* **Automated SELFIES data preprocessing pipeline**
* **Comprehensive evaluation metrics** for quality and diversity

---

## ⚙️ Setup and Installation

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

Ensure you have `requirements.txt` in the root directory, then run:

```bash
pip install -r requirements.txt
pip install selfies matplotlib rdkit-pypi
```

> **Note:** Ensure PyTorch version is compatible with your hardware (CPU/CUDA).

---

## 🗁 Data Preparation Workflow

This process converts SMILES data into SELFIES, builds the vocabulary, and generates SciBERT embeddings.

### Prerequisite

Place your data inside:

```
datasets/SMILES/
```

Each `.tsv` file should contain `CID`, `SMILES`, and `description` columns:

```
train.txt
validation.txt
test.txt
```

### Step 2.1 — Convert SMILES → SELFIES & Build Vocabulary

```bash
cd improved-diffusion/scripts/
python convert_smiles_to_selfies.py
```

**Output:**

```
datasets/SELFIES/
├── train.txt
├── test.txt
├── validation.txt
└── selfies_vocab.txt
```

### Step 2.2 — Clean SELFIES Dataset

```bash
python clean_selfies_dataset.py
```

**Output:** Cleaned `.txt` files in `datasets/SELFIES/`.

### Step 2.3 — Generate SciBERT Text Embeddings

```bash
python process_text_selfies.py -i train_val_256
python process_text_selfies.py -i validation_256
python process_text_selfies.py -i test
```

**Output:** `.pt` embedding files in `datasets/SELFIES/`.

---

## Training the Model

```bash
cd improved-diffusion/scripts/
python train_selfies.py
```

Checkpoints and logs are stored in `checkpoints/`.

### Optional: Train with Corruption/Masking

```bash
python train_correct_withmask_selfies.py
```

Used for post-sample repair.

---

## Generating Molecules (Sampling)

```bash
python text_sample_selfies.py \
--model_path ../../checkpoints/<your_model_checkpoint.pt> \
--num_samples 10000 \
--output_file ../../selfies_generation_results.txt
```

**Arguments:**

* `--model_path`: Path to trained model (e.g., `ema_0.9999_200000.pt`)
* `--num_samples`: Number of molecules to generate (10k–30k recommended)
* `--output_file`: Destination for generated SELFIES & SMILES

---

## Evaluating Generation Results

```bash
python evaluate_selfies_generation.py \
--generated ../../selfies_generation_results.txt \
--ground_truth ../../datasets/SELFIES/test.txt \
--output_dir ./evaluation_results
```

**Outputs:**

* Console report of all metrics
* `evaluation_results/evaluation_metrics.json` (Validity, Uniqueness, Novelty, Lipinski Ro5, etc.)
* `evaluation_results/plots/` with:

  * `property_distributions.png`
  * `metrics_summary.png`

---

## (Optional) Post-Processing / Repair

If trained with corruption, fix invalid SELFIES using:

```bash
python post_sample_selfies.py \
--model_path ../../correction_checkpoints/<your_correction_model.pt> \
--input_file ../../selfies_generation_results_invalid.txt \
--output_file ../../selfies_repaired_results.txt
```

---

## Key Project Files (SELFIES Migration)

### Core Logic

* `mytokenizers_selfies.py` – SELFIES tokenizer (`SELFIESTokenizer`)
* `mydatasets_selfies.py` – Custom ChEBI dataset loader

### Data Pipeline

* `convert_smiles_to_selfies.py` – SMILES → SELFIES conversion
* `clean_selfies_dataset.py` – SELFIES validation and cleanup
* `process_text_selfies.py` – SciBERT embedding generator

### Model Execution

* `train_selfies.py` – Training script
* `train_correct_withmask_selfies.py` – Corruption/masking variant
* `text_sample_selfies.py` – Sampling script
* `evaluate_selfies_generation.py` – Evaluation metrics
* `post_sample_selfies.py` – Invalid sample repair

---

## Acknowledgements

This project is based on the **Text-Guided Mask Denoising Language Model (TGM-DLM)** available at [https://github.com/Deno-V/tgm-dlm.git](https://github.com/Deno-V/tgm-dlm.git).
It has been extended to **SELFIES-based generation**, ensuring 100% valid molecules and robust text-conditioned diffusion modeling.

---

