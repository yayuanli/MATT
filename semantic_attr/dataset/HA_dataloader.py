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

import pandas as pd

class HADataset(Dataset):

    def video_loader(self, video_id, sf, ef):
        frames_dir = os.path.join(self.args.root1, str(video_id), "Export_py", "video_frames")

        if not os.path.exists(frames_dir):
            print(f"ERROR: VIDEO {video_id} NOT FOUND!")
            exit(1)

        max_ef = sum(1 for _ in os.scandir(frames_dir))

        if(ef > max_ef):
            ef = max_ef

        indices = np.linspace(sf, ef, self.args.clip_length, endpoint=True).astype(int)
        frames = indices.tolist()

        frames_tensor = torch.empty((self.args.clip_length, 3, 360, 640))
        
        i = 0
        
        for frame_num in frames:
            if not os.path.exists(os.path.join(frames_dir, "frame_" + str(frame_num).zfill(5) + ".jpg")):
                fp = os.path.join(frames_dir, "frame_" + str(frame_num).zfill(5) + ".jpg")
                raise FileNotFoundError(f"The file '{fp}' does not exist.")

            frame_tensor = torchvision.io.read_image(os.path.join(frames_dir, "frame_" + str(frame_num).zfill(5) + ".jpg"), mode= ImageReadMode.UNCHANGED)

            if(frame_tensor.shape != torch.Size([3, 360, 640])): 
                frame_tensor = self.crop_transform(frame_tensor)
            
            frames_tensor[i] = frame_tensor

            i += 1

        return frames_tensor

    def __init__(self, args, dataset_path):
        self.args = args
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

        # TODO: GET RID OF THIS:
        valid_indices = []
        for i in range(len(self.samples)):
            row = self.samples.iloc[i]
            video_id = row["video_id"]
            frames_dir = os.path.join(self.args.root1, str(video_id), "Export_py", "video_frames")

            if os.path.exists(frames_dir):
                valid_indices.append(i)
            else:
                print(f"Skipping missing video in init: {video_id}")

        self.samples = self.samples.iloc[valid_indices].reset_index(drop=True)

    def __getitem__(self, index):
        row = self.samples.iloc[index]

        label = row["label"]
        v = row["V"]
        arg1 = row["ARG1"]

        video_id = row["video_id"]
        sf = row["start_frame"]
        ef = row["end_frame"]

        frames = self.video_loader(video_id, int(sf) + 1, int(ef) + 1)
        frames = frames.permute(0, 2, 3, 1) # torch.Size([30, 3, 360, 640]) to torch.Size([30, 360, 640, 3])

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


        return frames, v, arg1, label_encoding

    def __len__(self):
        return self.samples.shape[0]