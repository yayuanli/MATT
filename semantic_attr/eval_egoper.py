from dataset.EgoPER_dataloader import EgoPERDataset
from model.multimodal_transformer import MultiModal_Transformer

import os 

import argparse
from huggingface_hub import hf_hub_download, scan_cache_dir
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, f1_score, recall_score
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

def get_args_parser():
    parser = argparse.ArgumentParser(description='Inference', add_help=False)
    parser.add_argument('--root1',
                        type=str,
                        required=True,
                        help='Path to dataset frames')
    parser.add_argument('--category',
                        default='all',
                        type=str,
                        help='Food category to evaluate for EgoPER. Choose from "coffee", "oatmeal", "pinwheels", "quesadilla", "tea", or "all"')
    parser.add_argument('--pretrain',
                        type=str,
                        required=True,
                        help='Dataset used for pretraining model. Select "ego4d" or "epic-kitchens"')
    parser.add_argument('--test_dataset_path',
                        type=str,
                        required=True,
                        help='Path to folder containing test xlsx files per food category')
    parser.add_argument('--test_size',
                        default=0,
                        type=int,
                        help='Size of test file will be recalculated during run time')
    parser.add_argument('--checkpoint_type',
                        default="all",
                        type=str,
                        help='Refers to type of checkpoint to evaluate. Choose "verb", "arg", "video", or "all".')
    parser.add_argument('--LaViLa_ckpt', default='./model/checkpoint_best.pt', type=str, help='Path to LaViLa checkpoint')
    parser.add_argument('--clip_length', default=30, type=int, help='Number of video frames sampled from each video clip')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for inference')
    parser.add_argument('--num_workers', default=24, type=int, help='Number of workers for data loading')
    return parser

def load_checkpoint(filepath, args, rank):
    checkpoint = torch.load(filepath)

    device = torch.device(f'cuda:{rank}')
    model = MultiModal_Transformer(args, rank).to(device=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def testing(args, model, test_dataloader, test_size, verb_metrics, arg_metrics, video_metrics, rank, result_file, df):
    device = torch.device(f'cuda:{rank}')

    model.eval()

    v_classifications = torch.zeros(test_size, dtype=torch.int32, device=device)
    arg_classifications = torch.zeros(test_size, dtype=torch.int32, device=device)
    video_classifications = torch.zeros(test_size, dtype=torch.int32, device=device)
    
    v_labels = torch.zeros(test_size, dtype=torch.int32, device=device)
    arg_labels = torch.zeros(test_size, dtype=torch.int32, device=device)
    video_labels = torch.zeros(test_size, dtype=torch.int32, device=device)

    multi_class = torch.zeros(test_size, dtype=torch.int32, device=device)

    idx = 0

    with torch.no_grad():
        for j, (frames, v, arg1, label) in enumerate(test_dataloader):
            frames = frames.to(device=f'cuda:{rank}')
            logits = model(frames, v, arg1, rank).to(device=f'cuda:{rank}')
            
            label = label.to(device=f'cuda:{rank}')
            label = label.int()

            v_actual = torch.max(label[:, 0, :], dim=1)[1]
            arg_actual = torch.max(label[:, 1, :], dim=1)[1]

            v_results = torch.max(logits[:, 0, :], dim=1)[1]
            arg_results = torch.max(logits[:, 1, :], dim=1)[1]

            #1 corresponsd to mistake/misalignement. 0 corresponds to correct/aligned
            batch_size = v_results.size(0)
            
            v_classifications[idx:idx + batch_size] = v_results  # Store verb predictions
            arg_classifications[idx:idx + batch_size] = arg_results  # Store argument predictions
            video_classifications[idx:idx + batch_size] = torch.logical_or(v_results, arg_results)  # Combined predictions
            
            v_labels[idx:idx + batch_size] = v_actual  # Store verb actual labels
            arg_labels[idx:idx + batch_size] = arg_actual  # Store argument actual labels
            video_labels[idx:idx + batch_size] = torch.logical_or(v_actual, arg_actual)  # Combined actual labels

            idx += batch_size

    multi_class[(v_classifications == 0) & (arg_classifications == 0)] = 0
    multi_class[(v_classifications == 1) & (arg_classifications == 0)] = 1
    multi_class[(v_classifications == 0) & (arg_classifications == 1)] = 2
    multi_class[(v_classifications == 1) & (arg_classifications == 1)] = 3
    
    multi_class = multi_class.cpu()

    df[f"{args.checkpoint_type}_preds"] = multi_class.numpy()

    for metric in verb_metrics: 
        metric.update(v_classifications, v_labels)
    for metric in arg_metrics:
        metric.update(arg_classifications, arg_labels)
    for metric in video_metrics:
        metric.update(video_classifications, video_labels)

    v = []
    arg = []
    video = []
    for metric in verb_metrics:
        v.append(metric.compute())
    for metric in arg_metrics:
        arg.append(metric.compute())
    for metric in video_metrics:
        video.append(metric.compute())

    v_classifications = v_classifications.tolist()
    arg_classifications = arg_classifications.tolist()
    video_classifications = video_classifications.tolist()
    
    v_labels = v_labels.tolist()
    arg_labels = arg_labels.tolist()
    video_labels = video_labels.tolist()

    v_balanced_accuracy = balanced_accuracy_score(v_labels, v_classifications)
    v_accuracy = accuracy_score(v_labels, v_classifications)
    v_precision = precision_score(v_labels, v_classifications)
    v_f1 = f1_score(v_labels, v_classifications)
    v_recall = recall_score(v_labels, v_classifications)

    arg_balanced_accuracy = balanced_accuracy_score(arg_labels, arg_classifications)
    arg_accuracy = accuracy_score(arg_labels, arg_classifications)
    arg_precision = precision_score(arg_labels, arg_classifications)
    arg_f1 = f1_score(arg_labels, arg_classifications)
    arg_recall = recall_score(arg_labels, arg_classifications)

    video_balanced_accuracy = balanced_accuracy_score(video_labels, video_classifications)
    video_accuracy = accuracy_score(video_labels, video_classifications)
    video_precision = precision_score(video_labels, video_classifications)
    video_f1 = f1_score(video_labels, video_classifications)
    video_recall = recall_score(video_labels, video_classifications)

    with open(result_file, 'a') as file:
        file.write("Torch metrics\n")
        file.write(f"Verb: Accuracy: {v[0]}, Precision: {v[1]}, Recall: {v[2]}, F1: {v[3]} \n")
        file.write(f"Arg: Accuracy: {arg[0]}, Precision: {arg[1]}, Recall: {arg[2]}, F1: {arg[3]} \n")
        file.write(f"Video: Accuracy: {video[0]}, Precision: {video[1]}, Recall: {video[2]}, F1: {video[3]} \n")
        file.write("\n")
        file.write("SkLearn Metrics \n")
        file.write(f"Verb: Balanced Accuracy: {v_balanced_accuracy} Accuracy: {v_accuracy} Precision: {v_precision} f1: {v_f1} recall: {v_recall} \n")
        file.write(f"Arg: Balanced Accuracy: {arg_balanced_accuracy} Accuracy: {arg_accuracy} Precision: {arg_precision} f1: {arg_f1} recall: {arg_recall} \n")
        file.write(f"Video: Balanced Accuracy: {video_balanced_accuracy} Accuracy: {video_accuracy} Precision: {video_precision} f1: {video_f1} recall: {video_recall} \n")

def evaluate_model(args, model, test_dataloader, rank, world_size, result_file, df):
    device = torch.device(f'cuda:{rank}')

    verb_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]
    arg_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]
    video_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]

    size_per_process = args.test_size // world_size

    for metric in verb_metrics:
        metric.reset()
    for metric in arg_metrics:
        metric.reset()
    for metric in video_metrics:
        metric.reset()

    testing(args, model, test_dataloader, size_per_process, verb_metrics, arg_metrics, video_metrics, rank, result_file, df)

def main(rank, world_size):
    parser = argparse.ArgumentParser('Testing multi-modal transformer', parents=[get_args_parser()])
    args = parser.parse_args()

    args.test_dataset_path = os.path.join(args.test_dataset_path, args.category, 'test.xlsx')
    df = pd.read_excel(args.test_dataset_path)
    args.test_size = len(df)

    ckpt_file_names = {
        "verb": "verb_model.pth",
        "arg": "arg_model.pth",
        "video": "video_model.pth"
    }
    eval_ckpt_types = ["verb", "arg", "video"] if args.checkpoint_type == "all" else [args.checkpoint_type]

    for ckpt_type in eval_ckpt_types:
        args.checkpoint_type = ckpt_type

        test_dataset = EgoPERDataset(args, args.test_dataset_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=args.num_workers, prefetch_factor=2, batch_size=args.batch_size, shuffle=False)
        hf_repo_id = f"mistakeattribution/egoper-finetuned-from-{args.pretrain}"
        checkpoint = hf_hub_download(repo_id=hf_repo_id, filename=ckpt_file_names[args.checkpoint_type])

        model = load_checkpoint(checkpoint, args, rank)

        checkpoint_path = Path(checkpoint).resolve()
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == hf_repo_id:
                for rev in repo.revisions:
                    if any(Path(f.file_path).resolve() == checkpoint_path for f in rev.files):
                        cache_info.delete_revisions(rev.commit_hash).execute()
                        break
                break

        os.makedirs("results/egoper", exist_ok=True)
        result_file = os.path.join("results", "egoper", f"{args.pretrain}_pretrained_{args.category}_{ckpt_type}.txt")

        evaluate_model(args, model, test_dataloader, rank, world_size, result_file, df)

        del model
        del test_dataloader
        del test_dataset
        torch.cuda.empty_cache()
    
    df.to_excel(f"./results/egoper/{args.pretrain}_pretrained_{args.category}_preds.xlsx") 

if __name__ == "__main__": 
    GPU_num = 0
    world_size = 1

    main(GPU_num, world_size)