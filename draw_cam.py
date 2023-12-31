"""
Overlay Gradcam predictions on xray images
------------------------------------------

python draw_cam.py -h
-----------------------
Author: Muhammad Faizan
"""

import argparse
import os
import sys
import torch
from pretrained_models import get_model
import utils
from utils import apply_mask, load_img
from grad_cam import GradCAM
import yaml
import warnings
import torchvision.transforms as T
from dataset.data import load_dataset
import random
import cv2
import numpy as np

# some command line arguments
def read_args():
    """
    Read command line arguments from the user
    ----------------------------

    """
    parser = argparse.ArgumentParser(prog = "Gradient localization on chest xray",
                                     description= "Overlay gradients activations on xray images")
    parser.add_argument("-i", "--image", type = str, default= None, help= "path to image")
    parser.add_argument("-l", "--label", type = str,
                        choices= ["covid_19", "lung_opacity", "normal", "pnuemonia"],
                        help="pick the label from the given choices if the label is not given")
    parser.add_argument("-m", "--model", type= str, 
                        choices = ["resnet18", "densenet121", "vgg16", "all"], 
                        help= "path to the model checkpoints from the given choice")
    parser.add_argument("-o", "--output", type = str, 
                        help = "output dir path")
    parser.add_argument('-c', '--config', type = str, 
                        help = "path to config file")
    parser.add_argument('--save', action= 'store_true', help= 'store a raw image')
    opts = parser.parse_args()
    return opts




if __name__ == "__main__":
    args = read_args()
    
    # pretrained checkpoins
    paths = {
        "vgg16": "weights/Runs/weights/lr3e-5_vgg16_cuda.pth",
        "densenet121": "weights/Runs/weights/lr3e-5_densenet121_cuda.pth",
        "resnet18": "weights/Runs/weights/lr3e-5_resnet18_cuda.pth"
    }
    #load a single example for inference.
    random.seed(42)
    data = load_dataset(config_file= args.config, batch_size= 1, 
                        kind = 'test')
    image, label = next(iter(data))

    # save the image if needed for comparsion purposes.
    

    path = paths[args.model]
    if not os.path.exists(path):
        raise Exception(
            f' {path} not found'
        )
        
    # load the model 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_info = torch.load(path, map_location= torch.device('cpu') if device == 'cpu' else None)
    model_state_dict = model_info["model_state_dict"]
    model = get_model(args.model, pretrained= False, num_classes=4, 
                      weights=model_state_dict)
    
    # model desired layer
    if args.model == 'vgg16' or args.model == 'densenet121':
        target_layer = model.features[-1]
    elif args.model == 'resnet18':
        target_layer = model.layer4[-1]

     # label to index mapping
    labels = {
        'covid_19': 0,
        'lung_opacity': 1,
        'normal': 2,
        'pneumonia': 3
    }
    idx_to_label = {v: k for k, v in labels.items()}
    
    if args.label is not None:
        label = args.label
    else:
        label = int(label)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # filter warninings
    warnings.filterwarnings("ignore", category= UserWarning)

    # use image for gradient based localization
    grad_cam = GradCAM(model = model, desired_layer= target_layer)
    label, mask = grad_cam(image, label)
    print(f'GradCAM created for label "{idx_to_label[label]}".')

    # unnormalize the image
    image = utils.unnormaliz_img(image)
    if args.save:
        img = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_BGR2RGB)
        utils.save_img(img, args.output + "/" + "raw_image.png")
    image = apply_mask(image, mask)

    # save the results
    utils.save_img(image, args.output + "/" + f"{args.model}_{idx_to_label[label]}_gradient_localizaton.png")

    


    
