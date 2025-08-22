import os
import torch
import argparse
from torch.utils.data import DataLoader
from src.utils import get_data
from src.train import train_model
from src.arch import MULTModel
from src.dataset import multimodal_collate_fn


parser = argparse.ArgumentParser(description='MInDS-14 Dataset for Intent Detection')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')

parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')

parser.add_argument('--dataset', type=str, default='minds14',
                    help='dataset to use (default: minds14)')

parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')


args = parser.parse_args()


torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())

valid_partial_mode = args.lonly + args.aonly
if valid_partial_mode == 0:
    args.lonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/a}only.")

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.set_default_dtype(torch.float32)


# -----Load the dataset ------


print("Start loading the data....")


# Ensure generator is CPU-based for reproducibility
#data shuffling in the loaders should be in the CPU as they are cpu tensors
g = torch.Generator(device="cpu").manual_seed(args.seed)

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'val')
test_data = get_data(args, dataset, 'test')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate_fn, generator=g)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate_fn, generator=g)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate_fn, generator=g)

print('Finish loading the data....')

#----------- Hyperparameters----------


hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)


output_dim_dict = {'minds14': 14}
criterion_dict = {'minds14': 'CrossEntropyLoss'}

hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')



# ------ building run name for experiment tracking -------

def build_run_name(hp):
    bits = [
        f"L{int(hp.lonly)}A{int(hp.aonly)}",
        f"layers{hp.nlevels}", f"heads{hp.num_heads}",
        f"lr{hp.lr}", f"bs{hp.batch_size}", f"seed{hp.seed}"
    ]
    return "_".join(map(str, bits))

hyp_params.name = build_run_name(hyp_params)


if __name__ == '__main__':
    model = MULTModel(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp_params.lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    settings = {
    "model": model,
    "optimizer": optimizer,
    "criterion": criterion,
    "scheduler": scheduler
}

    test_loss = train_model(settings, hyp_params, train_loader, valid_loader, test_loader)
