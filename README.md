# Explainable Multimodal Intent Detection for Task-oriented Dialogue Systems

## Description 

This project implements a **Multimodal Transformer (MulT)** based framework for **spoken language intent detection** using the [MInDS-14 dataset](https://huggingface.co/datasets/PolyAI/minds14).  
The goal is to compare **unimodal (text-only, audio-only)** and **bimodal (text+audio)** setups, and evaluate how accuracy, F1, and error metrics vary across modalities.
Additionaly, attention plots reveal the learnt modality alignment as well as feature importance on the classification

## Repository Structure
- extract_features.py
  - Extract and save text and audio features used downstream.
  - Key functions:
    - `extract_linguistic_features` - BERT Tokenizer to extract linguistic embeddings of shape (seq_len, 768).
    - `extract_acoustic_features` - produces Mel spectrograms for audio files of shape (seq_len, 80).
    - `process_and_save_dataset` -  performs data transformation where in feature vectors are saved in (minds14.pkl).
- prep_data.py
  - Prepare the MInDS-14 CSV dataset for processing (assemble English files, encode labels, split into train/test/val).
- main.py
  - Entry point for the model to initiate training
- modules/
  - Transformer architecture dependencies as provided by the original author
  - modules/multihead_attention.py 
  - modules/position_embedding.py
  - modules/transformers.py
- notebooks/ 
  - Simple visualizations of data explorations 
  - notebooks/feature_extract.ipynb - example runs of `extract_features.py` logic.
  - notebooks/eda.ipynb - exploratory data analysis.
- plots/ - saved training/experiment plots.
  - Training and Validation Losses against epochs curve
  - Variation of performance metrics (F1, accuracy across epochs)
- pretrained_models/ - saved model checkpoints (.pt).
- results/ - run logs / CSV outputs created by experiments.
- src/
  - src/utils.py - helper utilities for dataset loading and model save/load (`get_data`, `save_model`, `load_model`).
  - src/logger.py - `CSVLogger` used to log training metrics and plot history.
  - src/arch.py - `MULTModel`, the multimodal model architecture from original paper suited for this usecase
  - src/dataset.py - dataset class used to feed variable-length text/audio entries to the DataLoader.
  - src/train.py - runs training and evaluation loops 

## Basic usage (terminal) 

1. Clone the repository via https://github.com/saniaserrao/mmxai.git

2. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Prepare the CSV dataset (run once)
- Place raw MInDS-14 dataset under `data/MInDS-14` as expected by `prep_data.py`.
```powershell
python prep_data.py
# Writes combined CSV and label encoder for meta data storage (by default under data/processed_minds14, can be configured)
```

4. Extract features -- text + audio (run once)
```powershell
python extract_features.py
# Produces BERT embeddings, mel-spectrograms, and saves a pickled dataset (by default data/minds14_features/minds14.pkl)
```

5. Train / evaluate the model
```powershell
python main.py --dataset minds14 --data_path  data/minds14_features/ --batch_size 2  --num_epochs 10 --lr 1e-3 --nlevels 2 --num_heads 3  --lonly
# Edit CLI args or main.py defaults as needed.
# --lonly / --aonly run MuLT model in the unimodal case
# both flags provided or no flags provided runs the model in the bimodal case
```

## Acknowledgements

Model Architecture adopted from 

@inproceedings{tsai2019MULT,
  title={Multimodal Transformer for Unaligned Multimodal Language Sequences},
  author={Tsai, Yao-Hung Hubert and Bai, Shaojie and Liang, Paul Pu and Kolter, J. Zico and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month = {7},
  year={2019},
  address = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
}


