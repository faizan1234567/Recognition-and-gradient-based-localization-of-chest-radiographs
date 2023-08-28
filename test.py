"""
Check model performance on the trained checkpoints 
on test dataset or validation dataset


"""
import numpy as np
import matplotlib.pyplot as plt
from pretrained_models import get_model
from dataset.data import load_dataset
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             precision_score, 
                             recall_score)
import torch
import argparse
import logging
from train import get_num_correct, calculate_metrics
import yaml
from tqdm import tqdm
from time import sleep

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(filename= "logger_test.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type = int, default=32, help= "test loader batch size")
    parser.add_argument("--weights", type = str, help= "path to weights file")
    parser.add_argument("--model", type = str, default= "resnet18", help= "model name")
    parser.add_argument("--classes", type = int, default= 4, help= "number of classes")
    parser.add_argument("--config", type = str, default= "configs/configs.yaml", help = "configurations file")
    parser.add_argument("--kind", type = str, help= "inference type, i.e. val, test, train..")
    parser.add_argument("--subset", action= "store_true", help= "whether to use small subset")
    opt = parser.parse_args()
    return opt


def inference(batch: int = 32, 
              weights: str = "", 
              model: str = "resnet18",
              args: argparse.Namespace = None):
    """
    inference on the test set
    -------------------------

    args:
        batch: int
        weights: str
        model: str
    
    """
    # load the trained model 
    model = get_model(name = model, pretrained= False,
                      num_classes= args.classes, weights= weights)
    
    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
    
    # load the dataset
    data_loader = load_dataset(config_file= cfg, kind = args.kind, subset= args.subset)

    logger.info(f"Total samples in the dataset: {len(data_loader.dataset)}")

    if args.colab:
        cfg["general_configs"]["dataset splitted"] = "/gdrive/MyDrive/covid/data/COVID-19_Radiography_Dataset"
        cfg["DataLoader"]["num_workers"] = 2
    
    logger.info("Running inference on the specifed dataset.")
    print()
    
    # always use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    # run inference on the dataset.
    with tqdm(data_loader, unit= "batch") as tepoch:
        for (images, labels) in tepoch:
            sleep(0.01)
            

    

     


    