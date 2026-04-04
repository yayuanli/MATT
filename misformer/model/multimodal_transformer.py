from collections import OrderedDict
from lavila.models.utils import inflate_positional_embeds
from lavila.models import models
from lavila.utils.preprocess import generate_label_map, generate_tokenizer
from model.custom_encoders import text_encoder, video_encoder
from torch.nn import MultiheadAttention, TransformerEncoderLayer

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModal_Transformer(nn.Module):

    def init_lavila(self, args, rank):
        ckpt_path = os.path.join(args.LaViLa_ckpt)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        old_args = ckpt['args']

        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        lavila_model = getattr(models, old_args.model)(
            text_use_cls_token=old_args.use_cls_token,
            project_embed_dim=old_args.project_embed_dim,
            gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
            timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
            timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
            freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
            freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
            num_frames=args.clip_length,
            drop_path_rate=0,
        )
    
        if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
            # inflate weight
            print('=> inflating PE in models due to different frame numbers')
            state_dict = inflate_positional_embeds(
                lavila_model.state_dict(), state_dict,
                num_frames=args.clip_length,
                load_temporal_fix='bilinear',
        )
        torch.backends.cudnn.benchmark = True

        lavila_model.load_state_dict(state_dict, strict=True)
        lavila_model.to(device=f'cuda:{rank}')

        for param in lavila_model.parameters():
            param.requires_grad = False

        lavila_model.eval()

        self.lavila_model = lavila_model

        self.tokenizer =  generate_tokenizer(old_args.model)

    def __init__(self, args, rank: int, num_classes=2, d_model=256, nhead=8, num_decoder_layers=5, dim_feedforward=256, dropout=0.1):
        #Initialize the nn.Module parent class to properly initialize all the inherited attributes and functionalities
        super(MultiModal_Transformer, self).__init__()

        self.init_lavila(args, rank)
        
        # nhead is the number of parallel "attention heads" that focus on different relationships between the representations
        # dim_feedforward determines the hidden dimension within the network after the attention - this is the position-wise feed forward layer within the block
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout) #keep the dim_feedforward at 256
        
        # Stack multiple decoder layers to form the transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Classification layer to predict the alignment type (perfect, verb misaligned, etc.)
        # Returns the logits - aka unnormalized scores for each class. This must later be converted to probabilities using cross-entropy or softmax. Then use np.argmax to select the lass of the highest probability
        self.classification_head = nn.Linear(d_model, num_classes)
        
    def forward(self, frames, v, arg1, rank):
        # Assume text_encoding has shape [batch_size, 2, d_model]
        # and video_encoding has shape [batch_size, frames* patches, d_model]
        frames_encoding = video_encoder(self.lavila_model, frames, rank)

        v_encodings = []
        arg1_encodings = []

        # Loop through each string in the batch and encode
        for i in range(len(v)):
            # Encode each element in the tuple 'v' and 'arg1' using the text_encoder
            v_encoding = text_encoder(self.lavila_model, self.tokenizer, v[i], rank)
            arg1_encoding = text_encoder(self.lavila_model, self.tokenizer, arg1[i], rank)
            
            # Collect the encodings
            v_encodings.append(v_encoding.unsqueeze(0))  # unsqueeze to add batch dim
            arg1_encodings.append(arg1_encoding.unsqueeze(0))  # unsqueeze to add batch dim

        # Stack the encodings to form a batch tensor
        v_encodings = torch.cat(v_encodings, dim=0)  # Shape: [batch_size, embedding_dim]
        arg1_encodings = torch.cat(arg1_encodings, dim=0)  # Shape: [batch_size, embedding_dim]

        # Concatenate 'v' and 'arg1' encodings along a new dimension
        # Resulting shape will be [batch_size, 2, embedding_dim]
        text_encoding = torch.cat((v_encodings.unsqueeze(1), arg1_encodings.unsqueeze(1)), dim=1)
        text_encoding = text_encoding.squeeze(2) 

        #v_encoding = text_encoder(self.lavila_model, self.tokenizer, v)
        #arg1_encoding = text_encoder(self.lavila_model, self.tokenizer, arg1)
            
        #text_encoding = torch.cat((v_encoding.unsqueeze(1), arg1_encoding.unsqueeze(1)), dim=1)

        batch_size, semantic_roles, d_model = text_encoding.size()
        _ ,patches_x_frames,_ = frames_encoding.size()
        
        # Transpose for the transformer decoder: (seq_len, batch, d_model). According to the PyTorch documentation
        text_encoding = text_encoding.permute(1, 0, 2)
        video_encoding = frames_encoding.permute(1, 0, 2)

        text_encoding = text_encoding.to(device=f'cuda:{rank}')
        video_encoding = video_encoding.to(device=f'cuda:{rank}')

        tgt_attn_mask = torch.zeros(text_encoding.shape[0], text_encoding.shape[0]).to(device=f'cuda:{rank}')
        mem_attn_mask = torch.zeros(text_encoding.shape[0], video_encoding.shape[0]).to(device=f'cuda:{rank}')
        
        # Pass the text_encoding as query, and video_encoding as key & value to the Transformer decoder. Check what the output is
       
        decoder_output = self.transformer_decoder(tgt=text_encoding, memory=video_encoding, tgt_mask=tgt_attn_mask, memory_mask = mem_attn_mask)

        logits = self.classification_head(decoder_output)

        #Logits shape shape should be torch.Size([2, 1, 2]).

        logits = logits.permute(1, 0, 2) #Permute so the dimensons are [batch_size, semantic roles, 2 (aligned or misaligned)]
        #Shape of logits is [batch_size, 2, 2]

        return logits