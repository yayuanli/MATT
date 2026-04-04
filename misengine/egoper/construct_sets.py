# Usage: python construct_sets.py --input annotation.xlsx --egoper_dir /path/to/EgoPER --output_dir /path/to/output [--output annotation_final.xlsx] [--device cuda:0]
from allennlp.predictors.predictor import Predictor
import torch
import pandas as pd
import numpy as np
import math
import os
import argparse

class SRLPredictor:
    def __init__(self, device) -> None:
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", 
                                             cuda_device=0 if device == "cuda" else -1)
    def predict(self, sentences):
        instances = [{"sentence": str(sentence)} for sentence in sentences]
        results = self.predictor.predict_batch_json(instances)
        return results
    
def extract_argument(input_string, argument):
    search_pattern = f'{argument}'
    start_idx = input_string.find(search_pattern)
    if start_idx == -1:
        return None

    start_idx += len(search_pattern)
    end_idx = input_string.find(']', start_idx)
    if end_idx == -1:
        return input_string[start_idx:]
    
    return input_string[start_idx:end_idx].strip()

def constructSet(df, food, split, egoper_dir, output_dir):
    split_file = os.path.join(egoper_dir, food, f"{split}.txt")
    videos = []

    with open(split_file, 'r') as file:
        for line in file:
            videos.append(line.strip())

    filtered_df = df[df['video_id'].isin(videos)]
    
    directory_path = os.path.join(output_dir, food)
    os.makedirs(directory_path, exist_ok=True)
    filtered_df.to_excel(os.path.join(directory_path, f"{split}.xlsx"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SRL on EgoPER annotations, compute misalignment labels, and create per-food splits.")
    parser.add_argument('--input', type=str, default='annotation.xlsx', help='Input annotation xlsx')
    parser.add_argument('--egoper_dir', type=str, required=True, help='EgoPER data directory containing per-food split .txt files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for per-food split xlsx files')
    parser.add_argument('--output', type=str, default='annotation_final.xlsx', help='Output path for the final annotated xlsx')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device (default: cuda:0)')
    args = parser.parse_args()

    device = torch.device(args.device)
    srlpredictor = SRLPredictor(device=device)

    df = pd.read_excel(args.input)
    df = df[(df["action"] != "BG")]
    df = df[(df["action_type"] == "Normal") | (df["action_type"] == "Error_Slip") | (df["action_type"] == "Error_Modification")]

    action_sentences = df['action'].to_list()
    error_sentences = df['error_description'].to_list()

    action_SR = srlpredictor.predict(action_sentences)
    action_verbs = []
    action_args = []

    for i in range(len(action_SR)):
        verbs = action_SR[i]['verbs']
        if len(verbs) == 0:
            action_verbs.append("None")
            action_args.append("None")
        else: 
            action_verbs.append(verbs[0]['verb'])
            action_args.append(extract_argument(verbs[0]['description'], 'ARG1: '))
    
    error_SR = srlpredictor.predict(error_sentences)
    error_verbs = []
    error_args = []

    for i in range(len(error_SR)):
        verbs = error_SR[i]['verbs']
        if len(verbs) == 0:
            error_verbs.append("None")
            error_args.append("None")
        else: 
            error_verbs.append(verbs[0]['verb'])
            error_args.append(extract_argument(verbs[0]['description'], 'ARG1: '))
    
    df['V'] = action_verbs
    df['ARG1'] = action_args
    df['Error_V'] = error_verbs
    df['Error_Arg1'] = error_args

    df.replace('None', np.nan, inplace=True)
    df = df.dropna()

    labels = []

    for index, row in df.iterrows():
        action_v = row['V']
        action_arg = row['ARG1']
        error_v = row['Error_V']
        error_arg = row['Error_Arg1']

        if(action_v == error_v and action_arg == error_arg):
            labels.append(0)
        elif(action_v != error_v and action_arg == error_arg):
            labels.append(1)
        elif(action_v == error_v and action_arg != error_arg):
            labels.append(2)
        elif(action_v != error_v and action_arg != error_arg):
            labels.append(3)
        else: 
            labels.append(-1)
            print("INCORRECT LABELING")

    df['label'] = labels

    start_frame = []
    end_frame = []
    timestamps = df['timestamp']

    for t in timestamps:
        start, end = t.strip('[]').split(', ')

        start = int(math.floor(float(start) * 15))
        end = int(math.floor(float(end) * 15))

        start_frame.append(start)
        end_frame.append(end)
    
    df['start_frame'] = start_frame
    df['end_frame'] = end_frame

    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df.to_excel(args.output)
    
    for food in ["coffee", "oatmeal", "pinwheels", "quesadilla", "tea"]:
        for split in ["training", "validation", "test"]:
            constructSet(df, food, split, args.egoper_dir, args.output_dir)
