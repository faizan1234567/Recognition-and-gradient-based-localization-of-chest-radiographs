"""
Overlay Gradcam predictions on xray images
------------------------------------------

python draw_cam.py -h
---------------------
"""
# packages
import argparse
import os
import sys
import torch
from pretrained_models import get_model
from utils import apply_mask
from grad_cam import GradCAM

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
        
     # label to index map
    label = {
        'covid_19': 0,
        'lung_opacity': 1,
        'normal': 2,
        'pneumonia': 3
    }

    


    
