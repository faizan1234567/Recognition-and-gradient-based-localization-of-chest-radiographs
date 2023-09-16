"""
Overlay Gradcam predictions on xray images
------------------------------------------

python draw_cam.py -h
-----------------------
Author: Muhammad Faizan
"""
# packages
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

# some command line arguments
def read_args():
    """
    Read command line arguments..
    ----------------------------

    """
    parser = argparse.ArgumentParser(prog = "Gradient localization on chest xray",
                                     description= "Overlay gradients activations on xray images")
    parser.add_argument("-i", "--image", type = str, default= None, help= "path to image")
    parser.add_argument("-l", "--label", type = str,
                        choices= ["covid_19", "lung_opacity", "normal", "pnuemonia"],
                        help="pick the label from the given choices if the label is not given")
    parser.add_argument("-m", "--model", type= str, 
                        choices = ["resnet18", "densenet121", "vgg16"], 
                        help= "path to the model checkpoints from the given choice")
    parser.add_argument("-o", "--output", type = str, 
                        help = "output dir path")
    parser.add_argument('-c', '--config', type = str, 
                        help = "path to config file")
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
    
    # pick the model 
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
        label = labels[args.img.split('/')[-2]]

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # load the image
    image = load_img(path = path, cfg= cfg)
    warnings.filterwarnings("ignore", category= UserWarning)

    # use image for gradient based localization
    grad_cam = GradCAM(model = model, desired_layer= target_layer)
    label, mask = grad_cam(image, label)
    print(f'GradCAM created for label "{idx_to_label[label]}".')

    # unnormalize the image
    image = utils.unnormaliz_img(image)
    image = apply_mask(image, mask)

    # save the results
    utils.save_img(image, args.output + "/" + "gradient_localizaton.png")

    


    
