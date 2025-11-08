Text-Guided Diffusion Model for SELFIES Molecule Generation
This project implements a text-guided diffusion model for generating novel molecules, based on the tgm-dlm repository. This version has been significantly modified to operate using the SELFIES (SELF-referencIng Embedded Strings) molecular representation instead of SMILES.

This approach ensures 100% syntactic validity of all generated molecules by construction, eliminating the need for post-processing and validity checks that are common with SMILES-based models.

The model uses SciBERT to create text embeddings from molecular descriptions, which then guide the diffusion model to generate molecules matching the desired properties.

1. Setup and Installation
Create a Virtual Environment:

Bash

python3 -m venv venv
source venv/bin/activate
Install Dependencies: This project uses several libraries. Install them using the provided requiremts.txt file and add selfies and matplotlib.

Bash

pip install -r requiremts.txt
pip install selfies matplotlib rdkit-pypi
(Note: Ensure you have a compatible version of PyTorch installed for your hardware (CPU or CUDA).)

2. Data Preparation Workflow
This workflow converts the original SMILES dataset into a SELFIES dataset, builds the vocabulary, and generates the necessary text embeddings.

Prerequisite: Place your raw data files (train.txt, validation.txt, test.txt) inside the datasets/SMILES/ directory. These files should be tab-separated (.tsv) with CID, SMILES, and description columns.

Bash

# Navigate to the main scripts directory
cd improved-diffusion/scripts/
Step 2.1: Convert SMILES to SELFIES & Build Vocabulary This script reads from datasets/SMILES/, converts all SMILES to SELFIES, builds a complete vocabulary from all splits, and saves the new dataset to datasets/SELFIES/.

Bash

python convert_smiles_to_selfies.py
Output: datasets/SELFIES/ (containing train.txt, test.txt, validation.txt, and selfies_vocab.txt).

Step 2.2: Clean the SELFIES Dataset This step validates all SELFIES strings and removes any malformed entries (e.g., those from conversion errors or bad data) to prevent errors during training.

Bash

python clean_selfies_dataset.py
Output: Cleaned .txt files in datasets/SELFIES/ (backups of originals are created).

Step 2.3: Generate Text Embeddings This script uses the local SciBERT model (../../scibert/) to create description embeddings for each split. This must be done after creating the SELFIES dataset.

Bash

# Process the training split
python process_text_selfies.py -i train_val_256

# Process the validation split
python process_text_selfies.py -i validation_256

# Process the test split
python process_text_selfies.py -i test
Output: .pt files (e.g., train_val_256_scibert_desc.pt) in datasets/SELFIES/.

3. Training the Model
Once the data is prepared, you can start training the SELFIES diffusion model.

Bash

# Still inside improved-diffusion/scripts/
python train_selfies.py
Model checkpoints and logs will be saved to the checkpoints/ directory by default.

(Optional) Train with Corruption: To train with the corruption/masking policy (as in train_correct_withmask.py), use:

Bash

python train_correct_withmask_selfies.py
This model can be used for post-sample repair (Step 6).

4. Generating Molecules (Sampling)
Use the text_sample_selfies.py script to generate new molecules using a trained checkpoint. This script samples molecules based on the descriptions in the test set.

Bash

python text_sample_selfies.py \
    --model_path ../../checkpoints/<your_model_checkpoint.pt> \
    --num_samples 10000 \
    --output_file ../../selfies_generation_results.txt
--model_path: Path to your trained model checkpoint (e.g., ema_0.9999_200000.pt).

--num_samples: Number of molecules to generate. For robust evaluation, 10k-30k is recommended.

--output_file: Path to save the generated SELFIES and SMILES.

5. Evaluating Generation Results
Use the evaluate_selfies_generation.py script to calculate research-grade metrics for your generated molecules.

Bash

python evaluate_selfies_generation.py \
    --generated ../../selfies_generation_results.txt \
    --ground_truth ../../datasets/SELFIES/test.txt \
    --output_dir ./evaluation_results
--generated: The output file from Step 4.

--ground_truth: The test set to compare against for novelty and similarity.

--output_dir: A folder to save all results.

Evaluation Output:
Console: A detailed report of all metrics.

evaluation_results/evaluation_metrics.json: A JSON file with all computed metrics (Validity, Uniqueness, Novelty, Lipinski Ro5, Property Statistics, Internal/External Diversity, etc.).

evaluation_results/plots/: A directory containing publication-ready plots:

property_distributions.png: Histograms comparing properties (MW, LogP, QED, etc.) of generated vs. ground truth molecules.

metrics_summary.png: A bar chart of key metrics (Validity, Uniqueness, Novelty).

6. (Optional) Post-processing / Repair
If you trained a correction model (Step 3, optional) and your main model generated some syntactically invalid SELFIES (e.g., from selfies_generation_results_invalid.txt), you can repair them.

Bash

python post_sample_selfies.py \
    --model_path ../../correction_checkpoints/<your_correction_model.pt> \
    --input_file ../../selfies_generation_results_invalid.txt \
    --output_file ../../selfies_repaired_results.txt
Key Project Files (SELFIES Migration)
The core logic for the SELFIES migration is contained within improved-diffusion/scripts/:

Core Logic:

mytokenizers_selfies.py: The SELFIES tokenizer class (SELFIESTokenizer).

mydatasets_selfies.py: The custom ChEBIdataset loader for SELFIES.

Data Pipeline:

convert_smiles_to_selfies.py: Converts SMILES dataset to SELFIES and builds vocab.

clean_selfies_dataset.py: Validates and cleans the SELFIES dataset.

process_text_selfies.py: Generates SciBERT embeddings for descriptions.

Model Execution:

train_selfies.py: Main training script.

train_correct_withmask_selfies.py: Training script with corruption.

text_sample_selfies.py: Molecule generation/sampling script.

Evaluation:

evaluate_selfies_generation.py: Comprehensive metrics calculation.

quick_evaluate.py: A lightweight version for fast checks.

post_sample_selfies.py: Script for repairing malformed sequences.

Acknowledgements
This work is a modification of the original Text-Guided-Mask-Denoising-Language-Model, available at: https://github.com/Deno-V/tgm-dlm.git. The primary modification involves adapting the model from SMILES to SELFIES representations.
