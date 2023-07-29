import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, densenet121, vgg16

def create_model(architecture, num_classes = 2):
    """
    create a pretrained model.
    
    Parameters
    ----------
    model: torchvision.models
    """
    model = architecture(weights = "IMAGENET1K_V1")
    #freeze the model
    for params in model.parameters():
        params.requires_grad = False
        
    # customize last layer as per the need - number of classes in the dataset
    #ResNet18
    if type(model).__name__ == "ResNet":
        num_inftrs = model.fc.in_features 
        model.fc = nn.Linear(num_inftrs, num_classes)
    # VGG 16
    elif type(model).__name__ == "VGG":
        num_inftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_inftrs, num_classes)
    # DenseNet121
    elif type(model).__name__ == "DenseNet":
        num_inftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_inftrs, num_classes)
    return model

def get_model(name = "resnet18", pretrained = True, num_classes = 2):
    """
    Get state of art DenseNet model pretrained on imageNet dataset.

    Parameters
    ----------
    name: str
    pretrained: bool
    num_classes: int

    """
    if name == "resnet18":
        model = create_model(resnet18, num_classes= num_classes)
    elif name == "densenet121":
        model = create_model(densenet121, num_classes= num_classes)
    elif name == "vgg16":
        model = create_model(vgg16, num_classes= num_classes)
    else:
        print("Typo in model name or model is not available!")
        model = None
    return model


        