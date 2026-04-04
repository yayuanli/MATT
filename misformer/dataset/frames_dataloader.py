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