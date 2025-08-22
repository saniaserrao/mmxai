import torch
from torch import nn
import torch.nn.functional as F

from modules.transformers import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
         
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
        self.d_l, self.d_a = 30, 30

        self.lonly = hyp_params.lonly
        self.aonly = hyp_params.aonly

        # Transformer Encoder Hyperparameters
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        # Calculate the combined dimension based on the modalities used
        self.partial_mode = self.lonly + self.aonly
        if self.partial_mode == 1:
            # Unimodal case (text-only or audio-only)
            combined_dim = 2 * (self.d_l if self.lonly else self.d_a)
        else:
            # Bimodal case (text + audio)
            combined_dim = 2 * (self.d_l + self.d_a)
        
        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers for feature projection
        # reduce the original feature dimensions to a smaller, shared dimension (d_l, d_a)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            # (A) --> L
            # Transformer block where the language modality attends to the audio modality
            self.trans_l_with_a = self.get_network(self_type='la')
        if self.aonly:
            # (L) --> A
            # Transformer block where the audio modality attends to the language modality
            self.trans_a_with_l = self.get_network(self_type='al')
        
        # 3. Self Attentions (Memory)
        # Process the output of the cross-modal attention blocks to capture long-range dependencies within each modality.
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        
        # Projection layers for the final output
        # refinement of features in the linear layers with relu and dropout ,
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    
    # helper function that configures the transformer object based on modality case
    # if l/ a->l embedding space is set to linguistic embedding //ly for acoustic
    def get_network(self, self_type='l', layers=-1):

        # these are the cross-modal attention transformers that attend to features
        if self_type in ['l', 'al']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout 
        elif self_type in ['a', 'la']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a

        # memory networks receives the cross-modal output 
        # lang mem network stores info of language fused with other modalities
        # in  audio unimodal case, the lang net will work only with audio features
        # naming convention
        elif self_type == 'l_mem':
            # In the unimodal case, the memory network input dim is the output of the crossmodal
            embed_dim, attn_dropout = self.d_l if self.aonly else 2*self.d_l, self.attn_dropout
            if not self.aonly and self.lonly: # This is the l-only case
                embed_dim = 2*self.d_l
            elif self.aonly and not self.lonly: # This is the a-only case
                 embed_dim = 2*self.d_a
            elif self.aonly and self.lonly: # This is the multimodal case
                 embed_dim = 2*self.d_l
            else: # Bimodal
                 embed_dim = 2*self.d_l

        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a if self.lonly else 2*self.d_a, self.attn_dropout
            if not self.lonly and self.aonly: # This is the a-only case
                embed_dim = 2*self.d_a
            elif self.lonly and not self.aonly: # This is the l-only case
                 embed_dim = 2*self.d_l
            elif self.aonly and self.lonly: # This is the multimodal case
                 embed_dim = 2*self.d_a
            else: # Bimodal
                 embed_dim = 2*self.d_a
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
        
    def forward(self, x_l, x_a):
        
        """
        text and audio should have dimension [batch_size, seq_len, n_features]
        """
      
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)

        # Project the textual/audio features with a 1D convolution 
        # squashing down the dimensionality
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)

        # Transpose back to get [seq_len, batch_size, n_features] for the Transformer
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        last_h_l = None
        last_h_a = None
        
        if self.lonly and not self.aonly:
            # Unimodal L-only mode
            h_ls = self.trans_l_mem(torch.cat([proj_x_l, proj_x_l], dim=2))
            if type(h_ls) == tuple: h_ls = h_ls[0]
            last_h_l = h_ls[-1]
            last_hs = last_h_l
            
        elif self.aonly and not self.lonly:
            # Unimodal A-only mode
            h_as = self.trans_a_mem(torch.cat([proj_x_a, proj_x_a], dim=2))
            if type(h_as) == tuple: h_as = h_as[0]
            last_h_a = h_as[-1]
            last_hs = last_h_a
            
        else: # Bimodal mode (self.lonly and self.aonly are both True or both False)
            # (A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            h_ls = self.trans_l_mem(torch.cat([proj_x_l, h_l_with_as], dim=2)) # concat original + crossmodal
            if type(h_ls) == tuple: h_ls = h_ls[0]
            last_h_l = h_ls[-1]

            # (L) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_as = self.trans_a_mem(torch.cat([proj_x_a, h_a_with_ls], dim=2)) # concat original + crossmodal
            if type(h_as) == tuple: h_as = h_as[0]
            last_h_a = h_as[-1]
            
            last_hs = torch.cat([last_h_l, last_h_a], dim=1)
        
        # A residual block for final projection
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, last_hs