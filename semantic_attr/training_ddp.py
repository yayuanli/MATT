import os 

import argparse

import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, ConfusionMatrix

from model.multimodal_transformer import MultiModal_Transformer

from dataset.frames_dataloader import Ego4DDatasetTrain
from dataset.EgoPER_dataloader import EgoPERDataset
from dataset.EK_dataloader import EKDataset
from dataset.A101_dataloader import A101Dataset
from dataset.HA_dataloader import HADataset

import wandb

def setup_ddp(rank, world_size):

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

# Root for EgoPER is /nfs/turbo/coe-jjcorso1/aaditj/EgoPER_dataset/EgoPER_frames/
# Root for Assembly 101 is /nfs/turbo/coe-jjcorso1/aaditj/Assembly101/recordings/
# Root for HoloAsssist is /nfs/turbo/coe-jjcorso1/aaditj/HoloAssist/video_pitch_shifted/
def get_args_parser():
    parser = argparse.ArgumentParser(description='Training', add_help=False)
    parser.add_argument('--root1',
                        default='/nfs/turbo/coe-jjcorso1/aaditj/EgoPER_dataset/EgoPER_frames/',
                        type=str, help='path to dataset root')
    parser.add_argument('--root2',
                        default='',
                        type=str, help='path to dataset root.')
    parser.add_argument('--root3', 
                        default='', 
                        type=str, help='path to dataset root')
    # Ego4D, EgoPER, EK, Assembly101, HoloAssist
    parser.add_argument('--dataset', default='EgoPER', help="Dataset currently being utilized")
    parser.add_argument('--category', default='All/', type=str, help='Food category for EgoPER dataset. Relevant for EgoPER ONLY')
    parser.add_argument('--train_dataset_path', default='/nfs/turbo/coe-jjcorso1/aaditj/EgoPER_splits/', type=str, help='Food item and data split will be added at run time for EgoPER')
    parser.add_argument('--train_filename', default='alt_augment_training.xlsx', type=str, help='For EgoPER only, due to different categories') # ENSURE EITHER AUGMENTED OR NORMAL VERSION IS BEING USED
    parser.add_argument('--valid_dataset_path', default='/nfs/turbo/coe-jjcorso1/aaditj/EgoPER_splits/', type=str, help='Food item and data split will be added at run time for EgoPER')
    parser.add_argument('--valid_size', default=0, type=str, help='Will be reassigned')
    parser.add_argument('--test_dataset_path', default='/nfs/turbo/coe-jjcorso1/aaditj/EgoPER_splits/', type=str, help='Food item and data split will be added at run time for EgoPER. Say None if not using')
    parser.add_argument('--test_size', default=0, type=str, help='Will be reassigned')

    # Note these arguments do not work as expected in argparse. --test_as_valid and --use_f1 are True even if set to "False" at CLI
    parser.add_argument('--test_as_valid', default=True, type=bool, help='Whether to use test set as validation set')
    parser.add_argument('--use_f1', default=True, type=bool, help='APPLIES ONLY IF test_as_valid IS TRUE! Whether to use f1 or accuracy for saving checkpoints')
    
    parser.add_argument('--output_dir', default='/home/aaditj/LaViLa/Model_Parameters/EK_EgoPER/', type=str, help='Food name will be added at the end for EgoPEr')
    parser.add_argument('--recording_epochs', default='/home/aaditj/LaViLa/Model_Training/EK_EgoPER/log.txt', type=str, help='Text file to recording per-epoch metrics. Food name will be added for EgoPER')
    parser.add_argument('--LaViLa_ckpt', default='/home/aaditj/LaViLa/checkpoint_best.pt', type=str, help='')
    parser.add_argument('--pre_trained_ckpt', default='/home/aaditj/LaViLa/Model_Parameters/EK/video_model.pth', type=str, help='Say None if not being used') #/nfs/turbo/coe-jjcorso1/aaditj/Model_Parameters/Ego4D/~200k_samples/video_model_debug.pth
    parser.add_argument('--clip_length', default=30, type=int, help='')
    parser.add_argument('--global_batch_size', default=64, type=int, help='global = # of GPUs * batch size per GPU')
    parser.add_argument('--start_epoch', default=0, type=int, help='')
    parser.add_argument('--epochs', default=20, type=int, help='')
    parser.add_argument('--num_workers', default=12, type=int, help='')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='')
    parser.add_argument('--wandb_project', default='EgoPER_from_EK_pretraining', type=str, help='W&B project name')

    return parser
    
def train_one_epoch(model: torch.nn.Module, train_dataloader, optimizer: torch.optim.Optimizer, loss_fn, epoch: int, rank: int, world_size):
    
    model.train(True)

    device = torch.device(f'cuda:{rank}')
    
    acc = BinaryAccuracy().to(device)
    pre = BinaryPrecision().to(device)
    rec = BinaryRecall().to(device)
    f = BinaryF1Score().to(device)
    cm_train = ConfusionMatrix(task="binary", num_classes=2).to(device)

    batch_loss = 0.0
    num_batches = 0
    
    print(f"Training epoch #{epoch}")

    for i, (frames, v, arg1, label_encoding) in enumerate(train_dataloader):

        frames = frames.to(device=f'cuda:{rank}')
        labels = label_encoding.to(device=f'cuda:{rank}')

        optimizer.zero_grad()

        output = model(frames, v, arg1, rank).to(device=f'cuda:{rank}')

        loss = loss_fn(output, labels)

        loss_value = loss.item()
        batch_loss += loss_value
        num_batches += 1
        print(loss_value)

        v_actual = torch.max(labels[:, 0, :], dim=1)[1]
        arg_actual = torch.max(labels[:, 1, :], dim=1)[1]
        labels = torch.logical_or(v_actual, arg_actual).int()

        v_results = torch.max(output[:, 0, :], dim=1)[1]
        arg_results = torch.max(output[:, 1, :], dim=1)[1]
        preds = torch.logical_or(v_results, arg_results).int()

        acc.update(preds, labels)
        pre.update(preds, labels)
        rec.update(preds, labels)
        f.update(preds, labels)
        cm_train.update(preds, labels)

        loss.backward()
        optimizer.step()

    last_batch_loss_tensor = torch.tensor(loss_value, dtype=torch.float32, device=device)
    batch_loss_tensor = torch.tensor(batch_loss, dtype=torch.float32, device=device)
    num_batches_tensor = torch.tensor(num_batches, dtype=torch.float32, device=device)

    dist.all_reduce(last_batch_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(batch_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

    # Convert the last batch loss to a tensor
    avg_train_loss = batch_loss_tensor.item() / num_batches_tensor.item()

    # Calculate average loss across all processes
    avg_last_batch_loss = last_batch_loss_tensor.item() / world_size

    # Compute final values after synchronization
    final_accuracy = acc.compute().item()
    final_precision = pre.compute().item()
    final_recall = rec.compute().item()
    final_f1_score = f.compute().item()
    train_cm = cm_train.compute().cpu().numpy()

    # Reset metrics for the next epoch
    acc.reset()
    pre.reset()
    rec.reset()
    f.reset()
    cm_train.reset()

    if rank == 0:
        print(f"Epoch {epoch}, Average Last Batch Loss: {avg_last_batch_loss:.4f}")
        print(f"Accuracy: {final_accuracy:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1 Score: {final_f1_score:.4f}")
        # Log training metrics and confusion matrix to W&B
        wandb.log({
            "epoch": epoch,
            "Train/train_avg_loss": avg_train_loss,
            "Train/train_last_batch_loss": avg_last_batch_loss,
            "Train/train_accuracy": final_accuracy,
            "Train/train_precision": final_precision,
            "Train/train_recall": final_recall,
            "Train/train_f1": final_f1_score,
            "Train/train_TN": train_cm[0, 0],
            "Train/train_FP": train_cm[0, 1],
            "Train/train_FN": train_cm[1, 0],
            "Train/train_TP": train_cm[1, 1],
        }, step=epoch)

    return avg_last_batch_loss

def valid_one_epoch(args, model, valid_dataloader, valid_size, verb_metrics, arg_metrics, video_metrics, epoch, rank, loss):

    device = torch.device(f'cuda:{rank}')

    model.eval()

    v_classifications = torch.zeros(valid_size, dtype=torch.float32, device=device)
    arg_classifications = torch.zeros(valid_size, dtype=torch.float32, device=device)
    video_classifications = torch.zeros(valid_size, dtype=torch.float32, device=device)
    
    v_labels = torch.zeros(valid_size, dtype=torch.float32, device=device)
    arg_labels = torch.zeros(valid_size, dtype=torch.float32, device=device)
    video_labels = torch.zeros(valid_size, dtype=torch.float32, device=device)

    idx = 0

    # For confusion matrix tracking for each mode
    cm_verb = ConfusionMatrix(task="binary", num_classes=2).to(device)
    cm_arg = ConfusionMatrix(task="binary", num_classes=2).to(device)
    cm_video = ConfusionMatrix(task="binary", num_classes=2).to(device)

    for j, (frames, v, arg1, label) in enumerate(valid_dataloader):

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
        video_classifications[idx:idx + batch_size] = torch.logical_or(v_results, arg_results).int()  # Combined predictions
        
        v_labels[idx:idx + batch_size] = v_actual  # Store verb actual labels
        arg_labels[idx:idx + batch_size] = arg_actual  # Store argument actual labels
        video_labels[idx:idx + batch_size] = torch.logical_or(v_actual, arg_actual).int()  # Combined actual labels

        cm_verb.update(v_results, v_actual)
        cm_arg.update(arg_results, arg_actual)
        cm_video.update(torch.logical_or(v_results, arg_results).int(), torch.logical_or(v_actual, arg_actual).int())

        idx += batch_size

    for metric in verb_metrics:
        metric.update(v_classifications, v_labels)
    for metric in arg_metrics:
        metric.update(arg_classifications, arg_labels)
    for metric in video_metrics:
        metric.update(video_classifications, video_labels)

    dist.barrier()

    v = []
    arg = []
    video = []

    for metric in verb_metrics:
        v.append(metric.compute())
    for metric in arg_metrics:
        arg.append(metric.compute())
    for metric in video_metrics:
        video.append(metric.compute())

    # Get confusion matrix results. The matrix format is: [[TN, FP], [FN, TP]]
    verb_cm = cm_verb.compute().cpu().numpy()
    arg_cm = cm_arg.compute().cpu().numpy()
    video_cm = cm_video.compute().cpu().numpy()

    dist.barrier()

    cm_verb.reset()
    cm_arg.reset()
    cm_video.reset()

    if rank == 0:
        print(f"Epoch {epoch} accuracy - Verb: {v[0]}, Arg: {arg[0]}, Video: {video[0]}")
        file_name = args.recording_epochs
        with open(file_name, 'a') as file:
            file.write(f"Epoch: {epoch} \n")
            file.write(f"Loss during training: {loss} \n")
            file.write(f"Verb: Accuracy: {v[0]}, Precision: {v[1]}, Recall: {v[2]}, F1: {v[3]} \n")
            file.write(f"Arg: Accuracy: {arg[0]}, Precision: {arg[1]}, Recall: {arg[2]}, F1: {arg[3]} \n")
            file.write(f"Video: Accuracy: {video[0]}, Precision: {video[1]}, Recall: {video[2]}, F1: {video[3]} \n")
            file.write("\n")

        wandb.log({
            "epoch": epoch,
            "Valid/verb_accuracy": v[0],
            "Valid/verb_precision": v[1],
            "Valid/verb_recall": v[2],
            "Valid/verb_f1": v[3],
            "Valid/arg_accuracy": arg[0],
            "Valid/arg_precision": arg[1],
            "Valid/arg_recall": arg[2],
            "Valid/arg_f1": arg[3],
            "Valid/video_accuracy": video[0],
            "Valid/video_precision": video[1],
            "Valid/video_recall": video[2],
            "Valid/video_f1": video[3],
            "Valid/verb_TN": verb_cm[0, 0],
            "Valid/verb_FP": verb_cm[0, 1],
            "Valid/verb_FN": verb_cm[1, 0],
            "Valid/verb_TP": verb_cm[1, 1],
            "Valid/arg_TN": arg_cm[0, 0],
            "Valid/arg_FP": arg_cm[0, 1],
            "Valid/arg_FN": arg_cm[1, 0],
            "Valid/arg_TP": arg_cm[1, 1],
            "Valid/video_TN": video_cm[0, 0],
            "Valid/video_FP": video_cm[0, 1],
            "Valid/video_FN": video_cm[1, 0],
            "Valid/video_TP": video_cm[1, 1],
        }, step=epoch)

    dist.barrier()

    # Return verb, arg, and video accuracy on validation set
    return_val = []

    if args.use_f1 == True: 
        return_val.append(v[3])
        return_val.append(arg[3])
        return_val.append(video[3])
    else: 
        return_val.append(v[0])
        return_val.append(arg[0])
        return_val.append(video[0])
    

    return return_val

def test_one_epoch(args, model, test_dataloader, test_size, epoch, rank):
    device = torch.device(f'cuda:{rank}')

    model.eval()

    v_classifications = torch.zeros(test_size, dtype=torch.float32, device=device)
    arg_classifications = torch.zeros(test_size, dtype=torch.float32, device=device)
    video_classifications = torch.zeros(test_size, dtype=torch.float32, device=device)
    
    v_labels = torch.zeros(test_size, dtype=torch.float32, device=device)
    arg_labels = torch.zeros(test_size, dtype=torch.float32, device=device)
    video_labels = torch.zeros(test_size, dtype=torch.float32, device=device)

    idx = 0

    # For confusion matrix tracking for each mode
    verb_test_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]
    arg_test_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]
    video_test_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]

    cm_verb = ConfusionMatrix(task="binary", num_classes=2).to(device)
    cm_arg = ConfusionMatrix(task="binary", num_classes=2).to(device)
    cm_video = ConfusionMatrix(task="binary", num_classes=2).to(device)

    for j, (frames, v, arg1, label) in enumerate(test_dataloader):
        frames = frames.to(device=f'cuda:{rank}')
        logits = model(frames, v, arg1, rank).to(device=f'cuda:{rank}')
        
        label = label.to(device=f'cuda:{rank}')
        label = label.int()

        v_actual = torch.max(label[:, 0, :], dim=1)[1]
        arg_actual = torch.max(label[:, 1, :], dim=1)[1]

        v_results = torch.max(logits[:, 0, :], dim=1)[1]
        arg_results = torch.max(logits[:, 1, :], dim=1)[1]

        batch_size = v_results.size(0)
        
        v_classifications[idx:idx + batch_size] = v_results  # Store verb predictions
        arg_classifications[idx:idx + batch_size] = arg_results  # Store argument predictions
        video_classifications[idx:idx + batch_size] = torch.logical_or(v_results, arg_results).int()  # Combined predictions
        
        v_labels[idx:idx + batch_size] = v_actual  # Store verb actual labels
        arg_labels[idx:idx + batch_size] = arg_actual  # Store argument actual labels
        video_labels[idx:idx + batch_size] = torch.logical_or(v_actual, arg_actual).int()  # Combined actual labels

        cm_verb.update(v_results, v_actual)
        cm_arg.update(arg_results, arg_actual)
        cm_video.update(torch.logical_or(v_results, arg_results).int(), torch.logical_or(v_actual, arg_actual).int())

        idx += batch_size

    for metric in verb_test_metrics:
        metric.update(v_classifications, v_labels)
    for metric in arg_test_metrics:
        metric.update(arg_classifications, arg_labels)
    for metric in video_test_metrics:
        metric.update(video_classifications, video_labels)

    dist.barrier()

    v = []
    arg = []
    video = []

    for metric in verb_test_metrics:
        v.append(metric.compute())
    for metric in arg_test_metrics:
        arg.append(metric.compute())
    for metric in video_test_metrics:
        video.append(metric.compute())

    # Get confusion matrix results. The matrix format is: [[TN, FP], [FN, TP]]
    verb_cm = cm_verb.compute().cpu().numpy()
    arg_cm = cm_arg.compute().cpu().numpy()
    video_cm = cm_video.compute().cpu().numpy()

    dist.barrier()

    cm_verb.reset()
    cm_arg.reset()
    cm_video.reset()

    if rank == 0:
        wandb.log({
            "epoch": epoch,
            "Test/verb_accuracy": v[0],
            "Test/verb_precision": v[1],
            "Test/verb_recall": v[2],
            "Test/verb_f1": v[3],
            "Test/arg_accuracy": arg[0],
            "Test/arg_precision": arg[1],
            "Test/arg_recall": arg[2],
            "Test/arg_f1": arg[3],
            "Test/video_accuracy": video[0],
            "Test/video_precision": video[1],
            "Test/video_recall": video[2],
            "Test/video_f1": video[3],
            "Test/verb_TN": verb_cm[0, 0],
            "Test/verb_FP": verb_cm[0, 1],
            "Test/verb_FN": verb_cm[1, 0],
            "Test/verb_TP": verb_cm[1, 1],
            "Test/arg_TN": arg_cm[0, 0],
            "Test/arg_FP": arg_cm[0, 1],
            "Test/arg_FN": arg_cm[1, 0],
            "Test/arg_TP": arg_cm[1, 1],
            "Test/video_TN": video_cm[0, 0],
            "Test/video_FP": video_cm[0, 1],
            "Test/video_FN": video_cm[1, 0],
            "Test/video_TP": video_cm[1, 1],
        }, step=epoch)

    dist.barrier()

    # Return verb, arg, and video accuracy on validation set
    return_test = []

    if args.use_f1 == True: 
        return_test.append(v[3])
        return_test.append(arg[3])
        return_test.append(video[3])
    else: 
        return_test.append(v[0])
        return_test.append(arg[0])
        return_test.append(video[0])

    return return_test

def train_model(args, model, train_dataloader, valid_dataloader, test_dataloader, train_sampler, valid_sampler, test_sampler, rank, world_size):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) #Use Adam
    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device(f'cuda:{rank}')

    verb_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]
    arg_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]
    video_metrics = [BinaryAccuracy().to(device), BinaryPrecision().to(device), BinaryRecall().to(device), BinaryF1Score().to(device)]

    best_v = 0
    best_arg = 0
    best_video = 0

    valid_size_per_process = math.ceil(args.valid_size / world_size)
    test_size_per_process = 0

    if test_dataloader:
        test_size_per_process = math.ceil(args.test_size / world_size)

    for epoch in range(args.start_epoch, args.epochs):

        print(f"Starting epoch {epoch}")

        dist.barrier()

        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        if test_sampler:
            test_sampler.set_epoch(epoch)

        for metric in verb_metrics:
            metric.reset()
        for metric in arg_metrics:
            metric.reset()
        for metric in video_metrics:
            metric.reset()

        dist.barrier()

        loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch, rank, world_size)

        dist.barrier()

        test_stats = valid_one_epoch(args, model, valid_dataloader, valid_size_per_process, verb_metrics, arg_metrics, video_metrics, epoch, rank, loss)
        
        dist.barrier()

        if test_dataloader:
            if args.test_as_valid == True:
                test_stats = test_one_epoch(args, model, test_dataloader, test_size_per_process, epoch, rank)
            else:
                test_one_epoch(args, model, test_dataloader, test_size_per_process, epoch, rank)

        if rank == 0: 
            if test_stats[0] >= best_v: #Should I only save a checkpoint if its better than the best microAUC so far? 
                torch.save({ #Only save it if the accuracy is better, not Micro AUC
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, args.output_dir + '/verb_model.pth')
                best_v = test_stats[0]

            if test_stats[1] >= best_arg: #Should I only save a checkpoint if its better than the best microAUC so far? 
                torch.save({ #Only save it if the accuracy is better, not Micro AUC
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, args.output_dir + '/arg_model.pth')
                best_arg = test_stats[1]

            if test_stats[2] >= best_video: #Should I only save a checkpoint if its better than the best microAUC so far? 
                torch.save({ #Only save it if the accuracy is better, not Micro AUC
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, args.output_dir + '/video_model.pth')
                best_video = test_stats[2]
    
def main(rank, world_size):

    setup_ddp(rank, world_size)

    parser = argparse.ArgumentParser('Training multi-modal transformer', parents=[get_args_parser()])
    args = parser.parse_args()

    if rank == 0:
        wandb.init(project=args.wandb_project, config=vars(args))

    device = torch.device(f'cuda:{rank}')

    if args.dataset == 'EgoPER':

        args.train_dataset_path = args.train_dataset_path + args.category + args.train_filename
        args.valid_dataset_path = args.valid_dataset_path + '/' + args.category + 'validation.xlsx'

        if args.test_dataset_path != "None": 
            args.test_dataset_path = args.test_dataset_path + '/' + args.category + 'test.xlsx'
        
        args.output_dir = args.output_dir + args.category
        args.recording_epochs = args.recording_epochs + args.category.rstrip("/") + '.txt'

        train_dataset = EgoPERDataset(args, args.train_dataset_path)
        valid_dataset = EgoPERDataset(args, args.valid_dataset_path)
        
        if args.test_dataset_path == "None": 
            test_dataset = None
        else: 
            test_dataset = EgoPERDataset(args, args.test_dataset_path)

    elif args.dataset == 'Ego4D':

        train_dataset = Ego4DDatasetTrain(args, args.train_dataset_path)
        valid_dataset = Ego4DDatasetTrain(args, args.valid_dataset_path)
    
    elif args.dataset == 'EK':

        train_dataset = EKDataset(args, args.train_dataset_path)
        valid_dataset = EKDataset(args, args.valid_dataset_path)
    
    elif args.dataset == 'Assembly101':

        train_dataset = A101Dataset(args, args.train_dataset_path)
        valid_dataset = A101Dataset(args, args.valid_dataset_path)
    
    elif args.dataset == 'HoloAssist':

        train_dataset = HADataset(args, args.train_dataset_path)
        valid_dataset = HADataset(args, args.valid_dataset_path)

    args.valid_size = len(pd.read_excel(args.valid_dataset_path))

    args.test_size = -1

    if args.test_dataset_path != "None":
        args.test_size = len(pd.read_excel(args.test_dataset_path))

    dist.barrier()

    local_batch_size = args.global_batch_size // world_size

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader =  torch.utils.data.DataLoader(train_dataset, num_workers=args.num_workers, prefetch_factor=2, batch_size=local_batch_size, sampler=train_sampler)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    valid_dataloader =  torch.utils.data.DataLoader(valid_dataset, num_workers=args.num_workers, prefetch_factor=2, batch_size=local_batch_size, sampler=valid_sampler)
    
    test_sampler = None
    test_dataloader = None

    if args.test_dataset_path != 'None': 
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_dataloader =  torch.utils.data.DataLoader(test_dataset, num_workers=args.num_workers, prefetch_factor=2, batch_size=local_batch_size, sampler=test_sampler)

    model = MultiModal_Transformer(args, rank).to(device=device)

    if args.pre_trained_ckpt != 'None':
        checkpoint = torch.load(args.pre_trained_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    try:
        ddp_model = DDP(model, device_ids=[rank])
        print(f"Done, rank: {rank}")
    except Exception as e:
        print(f"Error creating DDP model: {e}")

    dist.barrier()

    if rank == 0:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    train_model(args, ddp_model, train_dataloader, valid_dataloader, test_dataloader, train_sampler, valid_sampler, test_sampler, rank, world_size)

    cleanup_ddp()

if __name__ == "__main__": 

    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"

    mp.spawn(main, args=(world_size,), nprocs = world_size, join=True)