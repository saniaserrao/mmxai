import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoTokenizer, AutoModel
import os
import pickle
from tqdm.auto import tqdm


''' design choice: the variable length of the text and audio sequences bmust be maintained for the downstream usecase
plotting attention weights. Therefore, padding and truncating the sequences have been avoided which 
has caused issues in batching.

BERT embeddings ---> [seq_len, hidden_size]-> [variable_length, 768]
Mel Specs ---> [frames, n_mels] ---> [variable_length, 80]
'''


def extract_linguistic_features(df, model_name="bert-base-uncased", batch_size=32
):
    print(f"Load BERT tokenizer and model :{model_name} --->")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if "text_translated" not in df.columns:
        raise ValueError(
            f"Data must contain a must contain a 'text_translated' column."
        )

    transcriptions = df["text_translated"].tolist()

    all_embeddings = []
    ids = list(range(len(transcriptions)))

    print(
        f"Processing {len(transcriptions)} transcriptions in batches of {batch_size} --->"
    )

    for i in tqdm(range(0, len(transcriptions), batch_size), desc="Encoding Progress"):
        batch_transcriptions = transcriptions[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        encoded_input = tokenizer(
            batch_transcriptions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_embeddings = (
                outputs.last_hidden_state
            )  # Shape: (batch_size, sequence_length, hidden_size)

        for j in range(batch_embeddings.shape[0]):
            non_pad_indices = torch.where(attention_mask[j] == 1)[0]
            unpadded_embedding = batch_embeddings[j, non_pad_indices, :]
            all_embeddings.append(unpadded_embedding.cpu().numpy())
            
    return all_embeddings
            

          
def extract_acoustic_features(df, sr=16000, n_mels=80, n_fft=400, hop_length=160
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length
    ).to(device)

    if "full_audio_path" not in df.columns:
        raise ValueError(
            f"Data must contain a 'full_audio_path' column."
        )

    audio_paths = df["full_audio_path"].tolist()
    ids = list(range(len(audio_paths)))

    all_mel_specs = []

    print(f"Processing {len(audio_paths)} audio files...")

    for i in tqdm(range(len(audio_paths)), desc="Extracting Mel Spectrograms"):
        audio = audio_paths[i]
        aid = ids[i]

        waveform, original_sr = torchaudio.load(audio)
        waveform = waveform.to(device)

        # processing the audio for uniformity

        # sr from 8Khz to 16KHz
        if original_sr != sr:
            resampler = T.Resample(orig_freq=original_sr, new_freq=sr).to(device)
            waveform = resampler(waveform)

        # stereo to mono conversion
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spec = mel_spec_transform(waveform)  # Shape: (channels, n_mels, frames)

        if mel_spec.shape[0] == 1:
            mel_spec = mel_spec.squeeze(0) # (n_mels, frames) removing 0th dim 
            
        mel_spec = mel_spec.transpose(0, 1) # (frames, n_mels) -> variable length, feature_dim
            
        all_mel_specs.append(mel_spec.cpu().numpy())

    return all_mel_specs

   

def process_and_save_dataset(input_csv_path, output_dir, sr=16000, hop_length=160):
    
    print(f"\n Starting Full Dataset Processing ---")
    df_combined = pd.read_csv(input_csv_path)
    
    final_dataset = {}
    splits = df_combined['split'].unique()

    for split in splits:
        print(f"\n--- Processing '{split}' split ---")
        df_split = df_combined[df_combined['split'] == split].reset_index(drop=True)

        text_features_list = extract_linguistic_features(df_split)
        audio_features_list = extract_acoustic_features(df_split)
        labels_list = df_split['intent_encoded'].tolist()
        item_ids_list = df_split.index.tolist()

        # dataset dict is of the form - {split: {text: ..., audio: ..., labels: ..., id: ...}}
        split_data = {
            'text': text_features_list,     # list of np.arrays (seq_len_i, feat_dim)
            'audio': audio_features_list,   # list of np.arrays (seq_len_i, feat_dim)
            'labels': labels_list,         
            'id': item_ids_list,          
            'sr': sr,
            'hop_length': hop_length
        }
        final_dataset[split] = split_data
        
        print(f"'{split}' split stored with {len(labels_list)} items")
        print(f"  Example text shape: {text_features_list[0].shape}")
        print(f"  Example audio shape: {audio_features_list[0].shape}")
        
        
    output_filename = "minds14.pkl"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(final_dataset, f)

    print(f"\nFinal combined dataset saved to: {output_path}")
    print("Processing complete!")

if __name__ == "__main__":
    data_dir= '../data/processed_minds14/intent_data.csv'
    output_dir = '../data/minds14_features'
    process_and_save_dataset(data_dir, output_dir, sr=16000, hop_length=160)
