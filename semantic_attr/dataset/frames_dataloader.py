import argparse
import os 

from collections import OrderedDict
import torchvision
from torchvision.io import ImageReadMode

import torch
from torch.utils.data import Dataset
from lavila.models import models
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from model.custom_encoders import text_encoder, video_encoder
import numpy as np
import random
from lavila.utils.preprocess import generate_label_map, generate_tokenizer
from model.multimodal_transformer import MultiModal_Transformer
import torch.nn.functional as F
from PIL import Image 
import torchvision.transforms as transforms


import decord
import pandas as pd

class Ego4DDataset(Dataset):

    def video_loader(self, clip1_uid, clip1_sf, clip1_ef, clip2_uid, clip2_sf, clip2_ef):
        start = int(clip1_sf) # + 1
        end = int(clip1_ef) # + 1

        if clip2_uid != "Not required": 
            end += int(clip2_ef) + 1

        indices = np.linspace(start, end - 1, self.args.clip_length, endpoint=True).astype(int)
        frames_1 = indices.tolist()

        if clip2_uid != "Not required": 
            frames_2 = [frame % (clip1_ef - 1) for frame in frames_1 if frame >= clip1_ef]
            frames_1 = [frame for frame in frames_1 if frame < clip1_ef]

        clip1_dir = ''
        clip2_dir = ''

        if clip1_uid + "_frames" in self.root1:
            clip1_dir = os.path.join(self.args.root1, clip1_uid + "_frames")
        elif clip1_uid + "_frames" in self.root2:
            clip1_dir = os.path.join(self.args.root2, clip1_uid + "_frames")
        elif clip1_uid + "_frames" in self.root3:
            clip1_dir = os.path.join(self.args.root3, clip1_uid + "_frames")
        else:
            print("ERROR: CLIP 1 NOT FOUND!")
            print(clip1_uid)
            exit(1)

        if clip2_uid != "Not required": 
            if clip2_uid + "_frames" in self.root1:
                clip2_dir = os.path.join(self.args.root1, clip2_uid + "_frames")
            elif clip2_uid + "_frames" in self.root2:
                clip2_dir = os.path.join(self.args.root2, clip2_uid + "_frames")
            elif clip2_uid + "_frames" in self.root3:
                clip2_dir = os.path.join(self.args.root3, clip2_uid + "_frames")
            else:
                print("ERROR: CLIP 2 NOT FOUND!")
                print(clip2_uid)
                exit(1)
            
        frames_tensor = torch.empty((self.args.clip_length, 3, 360, 640))
        i = 0
        for frame_num in frames_1:
            if not os.path.exists(os.path.join(clip1_dir, str(frame_num).zfill(5) + ".png")):
                raise FileNotFoundError(f"The file '{os.path.join(clip1_dir, str(frame_num).zfill(5) + '.png')}' does not exist.")
            
            frame_tensor = torchvision.io.read_image(os.path.join(clip1_dir, str(frame_num).zfill(5) + ".png"), mode= ImageReadMode.UNCHANGED)
            #frames1_paths.append(os.path.join(clip1_dir, str(frame_num).zfill(5) + ".png"))
            if(frame_tensor.shape != torch.Size([3, 360, 640])): 
                frame_tensor = self.crop_transform(frame_tensor)
            
            frames_tensor[i] = frame_tensor

            i += 1

        if clip2_dir: 
            for frame_num in frames_2:
                if not os.path.exists(os.path.join(clip2_dir, str(frame_num).zfill(5) + ".png")):
                    raise FileNotFoundError(f"The file '{os.path.join(clip2_dir, str(frame_num).zfill(5) + '.png')}' does not exist.")
                
                frame_tensor = torchvision.io.read_image(os.path.join(clip2_dir, str(frame_num).zfill(5) + ".png"), mode= ImageReadMode.UNCHANGED)
                #frames2_paths.append(os.path.join(clip2_dir, str(frame_num).zfill(5) + ".png"))
                if(frame_tensor.shape != torch.Size([3, 360, 640])): 
                    frame_tensor = self.crop_transform(frame_tensor)
                
                frames_tensor[i] = frame_tensor
                i += 1

        return frames_tensor


    def __init__(self, args, dataset_path):
        self.args = args
        self.root1 = set(os.listdir(self.args.root1))
        self.root2 = set(os.listdir(self.args.root2))
        self.root3 = set(os.listdir(self.args.root3))

        self.crop_transform = transforms.CenterCrop((360, 640))

        crop_size = 224

        self.transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]) #Confirmed
        ])

        self.label_mapping = None
        self.samples = pd.read_excel(dataset_path)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        # Narration = row["narration"]

        label = row["label"]
        v = row["V"]
        arg1 = row["ARG1"]

        clip1_uid = row["clip1_uid"] # Effectively the clip1 uid
        clip1_sf = row["clip1_start_frame"] # Effectively the clip1_start_frame
        clip1_ef = row["clip1_end_frame"] # Effectively the clip1_end_frame
        
        clip2_uid = row["clip2_uid"]
        clip2_sf = row["clip2_start_frame"]
        clip2_ef = row["clip2_end_frame"]
        
        # Run the video loader function, which returns a single, stacked tensor of frame tensors
        frames = self.video_loader(clip1_uid, clip1_sf, clip1_ef, clip2_uid, clip2_sf, clip2_ef)
        
        frames = frames.permute(0, 2, 3, 1) #torch.Size([30, 3, 360, 640]) to torch.Size([30, 360, 640, 3])

        # After we have the final narrations and frames
        if self.transform is not None:
            frames = self.transform(frames)

        # This is for the narration, not the label (aka the classification)!
        if self.label_mapping is not None:
            narration = self.label_mapping[narration]

        # Note on labels. First row of the matrix is for verb. Second row is for the argument. First column means aligned. SEcond column means misaligned. A 0 indicates false for that spot, and a 1 indicates true
        if(int(label) == 0):
            label_encoding = [[1, 0], [1, 0]]
            label_encoding = torch.tensor(label_encoding, dtype=torch.float32)
        elif(int(label) == 1): #means the verb is misaligned
            label_encoding = [[0, 1], [1, 0]]
            label_encoding = torch.tensor(label_encoding, dtype=torch.float32)
        elif(int(label) == 2):
            label_encoding = [[1, 0], [0, 1]]
            label_encoding = torch.tensor(label_encoding, dtype=torch.float32)
        elif(int(label) == 3):
            label_encoding = [[0, 1], [0, 1]]
            label_encoding = torch.tensor(label_encoding, dtype=torch.float32)
        else: 
            print("ERROR -> Incorrect Label!")

        return frames, v, arg1, label_encoding

    def __len__(self):
        return self.samples.shape[0]

'''
#For testing:

if __name__ == "__main__":
    #Arguments

    parser = argparse.ArgumentParser(description='LAVILA 0-shot evaluations', add_help=False)
    #args.add_argument('--dataset', default='egtea', type=str,
                        #choices=['ek100_cls', 'ek100_mir', 'charades_ego', 'egtea', 'ego4d_mcq'])
    parser.add_argument('--root',
                        default='/scratch1/home/yayuanli/dat/Ego4D/v2/v2/clips',
                        type=str, help='path to dataset root')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=16, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                       help='number of data loading workers per process')
    parser.add_argument('--clip-length', default=32, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')

    args = parser.parse_args()

    ckpt_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    old_args = ckpt['args']

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print('=> creating model: {}'.format(old_args.model))
    model = getattr(models, old_args.model)(
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
    model.to(device="cuda:0")
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=args.clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda:0")
    transformerModel = MultiModal_Transformer().to(device)


    dataset = Ego4DDatasetTrain(args, old_args)
    dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)

    #Print the batch size
    print(f"Length of Dataloader: {len(dataloader)}")
    for batch in dataloader:
        frames, v, arg1, label = batch
        print(f"Batch size: {len(frames)}")  # Should print the batch size
        #break
    
    
    for i, (frames, v, arg1, labels) in enumerate(dataloader):
        print(f"Batch {i + 1}:")

        #Print narrations and labels
        for j, (frames1, v_, arg1_, label1) in enumerate(zip(frames, v, arg1, labels)):
            print(f"   Sample {i + 1}:")
            print(f"   - Frames' size: {frames1.shape}")
            print(f"   - Verb: {v_}")
            print(f"   - Argument: {arg1_}")
            print(f"   - Label: {(label1)}")

            frames1 = frames1.unsqueeze(0)

            frames_encoded = video_encoder(model, frames1)
            tokenizer = generate_tokenizer(old_args.model)

            v_encoded = text_encoder(model, tokenizer, v)
            arg1_encoded = text_encoder(model, tokenizer, arg1)
            
            text_encoded = torch.cat((v_encoded.unsqueeze(1), arg1_encoded.unsqueeze(1)), dim=1)

            frames_encoded = frames_encoded.to(device)
            text_encoded = text_encoded.to(device)

            logits = transformerModel(text_encoded, frames_encoded)
            print("Logits Shape:", logits.shape)

            probabilities = F.softmax(logits, dim=1)
            c = torch.argmax(probabilities, dim=1)
            c_integer = c.item()
            
            
        if i == 2:  # Limit to first 3 batches for brevity
            break

    for epoch in range(2):  # Run for 2 epochs as an example
        print(f"Epoch {epoch+1}")
        for i, batch in enumerate(dataloader):
            print(f"Epoch {epoch+1}, Batch {i+1}")
            # Ensure data is on the correct device
            if torch.is_tensor(batch):
                print(f" - Device: {batch.device}")
'''
