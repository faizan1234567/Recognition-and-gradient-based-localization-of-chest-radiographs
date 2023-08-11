"""
Train on X ray classification task using different models such as 
- VGG16
- EfficientNet
- ResNet18

to train:
"python train.py -h"

Author: Muhammad Faizan
"""
import os
import sys
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import logging
import yaml
import mlflow
from tqdm import tqdm
import csv
# from PIL import image
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score)

import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from dataset import *
from pretrained_models import get_model
from dataset.data import load_dataset
from mlflow import log_metric, log_param, log_params, log_artifacts

# check if the parent directory is in the path if not then append it
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.app(str(ROOT))

# configuration file path
config_file = "configs/configs.yaml"

# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(filename= "logger.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# get command line arguments
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default= 100, help = "number of iterations")
    parser.add_argument("--learning_rate", type = float, default= 1e-4, help= "learning rate value")
    parser.add_argument("--batch", type = int, default=16, help= "batch size")
    parser.add_argument("--weight_decay", type = float, default=1e-5, help="value of weight decay parameter")
    parser.add_argument("--save", type = str, help= "path to runs directory to save the results")
    parser.add_argument("--workers", type = int, default=8, help= "number of data loader workers")
    parser.add_argument('--model', type = str, help= "select model from: resnet18, DenseNet121, vgg16")
    parser.add_argument('--colab', action= "store_true", help="colab training option")
    opt = parser.parse_args()
    return opt


# num corrects
def get_num_correct(preds, labels):
    """
    get num of corrects predictions

    Parameters
    ----------
    preds: torch.tensor
    labels: torch.tensor
    """
    return preds.argmax(dim=1).eq(labels).sum().item()

# calculte metrics
def calculate_metrics(y_pred, y_true, flag = "all"):
    """
    calculate metrics for the training logs
    Parameters
    ----------
    y_pred: torch.tensor
    y_true: torch.tensor
    flag: str
    """
    if flag == "all":
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1_score = f1_score(y_true, y_pred)
        return (accuracy, precision, recall, f1_score)
    else:
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

# train the machine laenring model
def train(model,
          optimizer, 
          criterion,
          schedular,
          train_loader,
          val_loader,
          args,
          val_every
          ):
    """
    train a model such vgg16, efficientNet, ResNet18
    model: torchvision.models
    optimizer: torch.optim.Adam
    criterion: torch.nn.BCELossWithLogits
    schedular: torch.optim.lr_schedular
    configs:str
    args: argparse.Namespace
    """

    # train a deep leanring model on image classfication task
    # create a directory to store training runs
    logger.info("creating a runs directory to store training runs...")
    if args.save:
        runs = os.path.join(args.save, 'Runs')
        weights = os.path.join(runs, 'weights')
        dirs = [runs, weights]
        if not os.path.exists(runs) and not os.path.exists(weights):
            for dir in dirs:
                os.makedirs(dir)
    
    #log training informations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Learning rate: {}, batch size: {}, epochs: {}, device: {}".format(args.learning_rate, args.batch, 
                                          args.epochs, device))
    # initialize training & validation variables 
    print()
    
    valid_loss_min = np.inf
    cols =  [
        'epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'
    ]
    rows = []

    # train and validation set size
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    
    
    # select training mode
    model.to(device)
    train_loop = tqdm(train_loader)

    # starting training.
    for epoch in range(args.epochs):
        epoch_loss = 0
        train_corrects = 0
        model.train()

        for (images, labels) in train_loop:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            epoch_loss += loss.item() * labels.size(0)
            train_corrects += get_num_correct(predictions, labels)

            train_loop.set_description(f'Epoch [{epoch+1:2d}/{epochs}]')
            train_loop.set_postfix(
                loss=loss.item(), acc=train_corrects/train_samples
            )

        # now log epoch performance 
        train_loss = epoch_loss/train_samples
        train_acc = train_corrects/train_samples
        schedular.step() 

        # print("Training: Epoch loss: {:.4f}, Epoch accuracy: {:.4f}".format(avg_iters_loss, avg_iters_acc), end = " | ")
        log_metric("Training_accuracy", train_acc)
        log_metric("Training_loss", train_loss)

        # validate the model
        if epoch % val_every == 0:
            model.eval()
            val_loss = 0
            val_corrects = 0
            with torch.no_grad():
                for (images, labels) in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    val_predictions = model(images)
                    val_iter_loss = criterion(val_predictions, labels.float())
                    
                    val_loss += val_iter_loss.item() * labels.size(0)
                    val_corrects += get_num_correct(predictions, labels)

                # average over the epoch
                avg_val_loss = val_loss/val_samples
                avg_val_acc = val_corrects / val_samples
                rows.append([epoch, train_loss, train_acc, avg_val_loss, avg_val_acc])
               
                log_metric("Validation_accuracy", avg_val_acc)
                log_metric("Validation_loss", avg_val_loss)

                # write loss and acc
                train_loop.write(
                f'\n\t\tAvg train loss: {train_loss:.6f}', end='\t'
            )
            train_loop.write(f'Avg valid loss: {avg_val_loss:.6f}\n')

            # save model if validation loss has decreased
            if avg_val_loss <= valid_loss_min:
                train_loop.write('\t\tvalid_loss decreased', end=' ')
                train_loop.write(f'({valid_loss_min:.6f} -> {avg_val_loss:.6f})')
                train_loop.write('\t\tsaving model...\n')
                torch.save(
                    model.state_dict(),
                    f'models/lr3e-5_{model_name}_{device}.pth'
                )
                valid_loss_min = avg_val_loss

    # write running results for plots
    with open(f'{runs}/{model_name}.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(cols)
        csv_writer.writerows(rows)

# run ..
if __name__ == "__main__":
    logger.info("Initializing..")
    # start mlflow tracking 
    mlflow.start_run()
    # open settings from a config file
    val_every = 1

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    # read commmand line args
    args = read_args()
    
    # data loader batch size
    if args.batch:
        batch = args.batch
    else:
        batch = cfg["DataLoader"]["batch_size"]
    
    # training epochs
    if args.epochs:
       epochs = args.epochs
    else:
        epochs = cfg["Training"]["epochs"]
    
    # optimizer learning rate
    if args.learning_rate:
        lr = args.learning_rate
    else:
        lr = cfg["Training"]["learning_rate"]
    
    # data loader workers
    if args.workers:
        workers = args.workers
    else:
        workers = cfg["DataLoader"]["workers"]
    
    # optimizer weigth decay
    if args.weight_decay:
        weight_decay = args.weight_decay
    else:
        weight_decay = cfg["Training"]["weight_decay"]
    
    # model selection
    if args.model:
        model_name = args.model
    else:
        model_name = cfg['Training']["model_name"]
    
    # set paths for colab drive dataset directory
    if args.colab:
        cfg["general_configs"]["dataset splitted"] = "/gdrive/MyDrive/covid/data/COVID-19_Radiography_Dataset"
        cfg["DataLoader"]["num_workers"] = 2
    
    model = get_model(model_name, pretrained= True,
                      num_classes=cfg["DataLoader"]["num_classes"])
    # get an optimizer
    optimizer = optim.Adam(model.parameters(), lr= lr)

    # get loss function
    loss_function = nn.CrossEntropyLoss()

    # leanring rate schedular
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.1)

    # training data loader
    training_loader = load_dataset(config_file= cfg, kind="train")

    #valiation data loader
    validation_loader = load_dataset(config_file= cfg, kind = 'val')
    
    # list of training configuration to change when needed.

    all_params = {"lr": lr,
                  "workers": workers,
                  "batch": batch,
                  "weight_decay": weight_decay,
                  "epochs": epochs,
                  "model_name": model_name,
                  "optimizer": "adam",
                  "loss": "CrossEntropyLoss",
                  "num_classes": "4",
                  "schedular_steps": 7,
                  "schedular_gamma": 0.1,
                  "val_every": val_every
                  }
    
    # log params configs
    log_params(all_params)
    logger.info("Starting Training")
    print()
    train(model =  model,
          optimizer= optimizer,
          criterion= loss_function,
          schedular= exp_lr_scheduler,
          val_every = val_every,
          train_loader= training_loader,
          val_loader= validation_loader,
          args= args)
    print()
    logger.info("Training finished.")
    # end mlflow tracking
    mlflow.end_run()    
    












