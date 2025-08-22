import torch
import os
from src.dataset import Multimodal_Datasets
from src.arch import MULTModel  




def get_data(args, dataset_name, split='train'):
    data_file = os.path.join(args.data_path, f"{dataset_name}.pkl")
    cache_file   = os.path.join(args.data_path, f"{dataset_name}_{split}.pt")
    meta_path = os.path.join(args.data_path, "label_encoder.pkl")


    if not os.path.exists(cache_file):
        print(f"  - Creating new {split} data")
        
        data = Multimodal_Datasets(data_file, split, dataset_name, meta_path=meta_path)
        torch.save(data, cache_file)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(cache_file, weights_only=False)
    return data



def save_model(model, name='mult', map_location=None):
    
    torch.save(model.state_dict(), f'pretrained_models/{name}.pt')


def load_model(name,hyp_params, map_location=None):
    model=MULTModel(hyp_params)
    state_dict = torch.load(f'pretrained_models/{name}.pt', map_location="cpu")
    model.load_state_dict(state_dict)
    return model
