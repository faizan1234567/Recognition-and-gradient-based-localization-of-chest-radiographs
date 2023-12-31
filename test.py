

"""
Check model performance on the trained checkpoints 
on test dataset or validation dataset

python test.py -h for more information
--------------------------------------
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
from tabulate import tabulate

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
    parser.add_argument("--colab", action= "store_true", help= "colab option")
    parser.add_argument("--manual_table", action="store_true", help="log manual table")
    opt = parser.parse_args()
    return opt

# run inference on the batches of images for testing.
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
    logger.info("loading a Trained model")
    model_info = torch.load(args.weights, map_location= torch.device("cpu"))
    epoch = model_info["epoch"]
    logger.info(f"Total trained epochs: {epoch}")
    model_sd = model_info["model_state_dict"]
    model = get_model(name = model, pretrained= False,
                      num_classes= args.classes, weights= model_sd)
    
    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
    
    # load the dataset
    data_loader = load_dataset(config_file= cfg, kind = args.kind, subset= args.subset, 
                               batch_size= args.batch)
    total_samples = len(data_loader.dataset)
    logger.info(f"Total samples in the dataset: {total_samples}")

    if args.colab:
        cfg["general_configs"]["dataset splitted"] = "/gdrive/MyDrive/covid/data/COVID-19_Radiography_Dataset"
        cfg["DataLoader"]["num_workers"] = 2
    
    logger.info("Running inference on the specifed dataset.")
    print()
    
    # select  hardware acceleration device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    val_corrects = 0
    loader = tqdm(data_loader)
    precision, recall, f1_score, accuracy= 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            loader.set_description(f'Inference')
            images, labels = images.to(device), labels.to(device)
            val_predictions = model(images)
            val_corrects += get_num_correct(val_predictions, labels)
            preds_classes = val_predictions.argmax(dim=1)
            accs, p, r, f1 = calculate_metrics(preds_classes, labels, 
                                                "all", average= "macro")
            precision += p
            recall += r
            f1_score += f1
            accuracy += accs
            loader.set_postfix(
                    acc=accs)

        # average over the epoch
        mean_precision = precision/len(data_loader)
        mean_recall = recall/len(data_loader)
        mean_f1 = f1_score/len(data_loader)
        accuracy = accuracy/len(data_loader)
    
    # print table 
    if args.manual_table:
        logger.info("Evaluation Results")
        logger.info("+-----------------------+---------+")
        logger.info("| Metric               |  Value   |")
        logger.info("+-----------------------+---------+")
        logger.info(f"| Precision macro      | {mean_precision: .3f}   |")
        logger.info(f"| Recall macro         | {mean_recall: .3f}   |")
        logger.info(f"| F1 Score macro       | {mean_f1: .3f}   |")
        logger.info(f"| Accuracy             | {accuracy: .3f}   |")
        logger.info("+-----------------------+---------+")
    else:
        print("Evaluation Results")
        table_data = [
        ["Precision macro", mean_precision],
        ["Recall macro", mean_recall],
        ["F1 Score macro", mean_f1],
        ["Accuracy", accuracy]]
        table_headers = ["Metric", "Value"]
        table = tabulate(table_data, headers=table_headers, tablefmt="grid", numalign="right", stralign="center")
        print(table)

def main():
    args = read_args()
    inference(args.batch, args.weights, args.model, args)

if __name__ == "__main__":
    main()
