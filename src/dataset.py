import numpy as np
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import torch
from torch.nn.utils.rnn import pad_sequence




if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    


def multimodal_collate_fn(batch):
    """
    Pads variable-length text and audio sequences in the batch.
    batch: list of (X, Y, META)
    """
    texts, audios, labels, metas = [], [], [], []

    for (x_text, x_audio), y, meta in batch:
        texts.append(x_text)
        audios.append(x_audio)
        labels.append(y)
        metas.append(meta)

    # pad_sequence: pad to longest in batch
    texts_padded = pad_sequence(texts, batch_first=True)   # [B, max_len_text, feat_dim]
    audios_padded = pad_sequence(audios, batch_first=True) # [B, max_len_audio, feat_dim]
    labels = torch.stack(labels)

    return (texts_padded, audios_padded), labels, metas

class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train', data_name='minds14', meta_path=None):
        super(Multimodal_Datasets, self).__init__()

        try:
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        
        
        if split_type not in dataset:
            raise ValueError(f"Split type '{split_type}' not found in the dataset dictionary.")

        self.text = [torch.tensor(x, dtype=torch.float32) for x in dataset[split_type]['text']]
        self.audio = [torch.tensor(x, dtype=torch.float32) for x in dataset[split_type]['audio']]
        self.labels = torch.tensor(dataset[split_type]['labels'], dtype=torch.long)

        

        # --- Other dataset information ---
        # Note: this is STILL a numpy array (for string IDs, etc.)
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type] else None
        
        
        if meta_path is not None:
            with open(meta_path, 'rb') as f:
                self.label_encoder: LabelEncoder = pickle.load(f)
        else:
            self.label_encoder = None
            
        # number of unique classes
        self.num_classes = int(self.labels.max().item()) + 1
        self.data_name = data_name
        self.n_modalities = 2 # text/audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self): 
        # Returns sequence lengths of the modalities
        return [x.shape[0] for x in self.text], [x.shape[0] for x in self.audio]

    def get_dim(self):
        # Returns feature dimensions of the modalities
        return self.text[0].shape[1], self.audio[0].shape[1]
    
    def get_num_classes(self):
        # Returns the total number of intent classes.
        return self.num_classes

    def get_lbl_info(self):
        # Modified for intent classification: returns the number of classes.
        # This is the crucial information for the model's output layer.
        return self.num_classes
        
        return self.num_classes
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X_text = self.text[index]
        X_audio = self.audio[index]
        
        X = (X_text, X_audio)
        Y = self.labels[index]
        
        if self.label_encoder is not None:
            label_str = self.label_encoder.inverse_transform([Y.item()])[0]
        else:
            label_str = str(Y.item())


        # Meta data handling - idx plus textual labels reference
        if self.meta is not None:
            current_meta = self.meta[index]
            # Assuming meta items are IDs, which could be tuples, lists, or single values
            if isinstance(current_meta, (list, tuple)):
                META = tuple(current_meta) + (label_str,)
            else:
                META = (current_meta, '', label_str) 
        else:
            META = (None, None, label_str)
        
        return X, Y, META