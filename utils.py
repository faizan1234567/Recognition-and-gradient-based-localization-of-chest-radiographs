"""
Utils functions
---------------

chest x ray recognition
Some of the plot utils have been borrowd 
from: https://github.com/priyavrat-misra/xrays-and-gradcam/blob/master/plot_utils.py


python utils.py
---------------
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import sklearn
import yaml
from PIL import Image
from dataset.data import load_dataset
from pretrained_models import get_model
from dataset.data import get_transforms
import pandas as pd
import cv2
import argparse

import torch

# device type
device = "cuda" if torch.cuda.is_available() else "cpu"

# get total number of predictions
def get_all_preds(model, loader):
    model.eval()
    with torch.no_grad():
        all_preds = torch.tensor([], device=device)
        all_labels = torch.tensor([], device=device)
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    return all_preds, all_labels

def get_confmat(targets, preds):
    stacked = torch.stack(
        (targets,
         preds.argmax(dim=1)), dim=1
    ).tolist()
    confmat = torch.zeros(4, 4, dtype=torch.int16)
    for t, p in stacked:
        confmat[int(t), int(p)] += 1

    return confmat

# get different results such as acc, precision, recall, and f1score
def get_results(confmat, classes):
    results = {}
    d = confmat.diagonal()
    for i, l in enumerate(classes):
        tp = d[i].item()
        tn = d.sum().item() - tp
        fp = confmat[i].sum().item() - tp
        fn = confmat[:, i].sum().item() - tp

        accuracy = (tp+tn)/(tp+tn+fp+fn)
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f1score = (2*precision*recall)/(precision+recall)

        results[l] = [accuracy, recall, precision, f1score]

    return results

# load an image
def load_img(path, cfg):
    img = Image.open(path)
    img = get_transforms(cfg, kind = "val")(img).unsqueeze(0)
    print(img.shape)
    return img

# unnormalize the image (reconstruct the orgin image)
def unnormaliz_img(img):
    image = img.cpu().numpy()
    image = np.squeeze(np.transpose(image[0], (1, 2, 0)))
    image = image * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))  # un-normalize
    image = image.clip(0, 1)
    return image

# save the image
def save_img(image, path):
    image = image[:, :, ::-1]  # RGB -> BGR
    image = Image.fromarray(image, 'RGB')
    image.save(path)  # saved as RGB
    print(f'GradCAM masked image saved to "{path}".')

# plot results such accuracies and losses
def plot_results(file):
    data = pd.read_csv(file)
    filename = file.split("/")[-1][:-4]

    plt.style.use("seaborn-whitegrid")
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (14,  4))
    
    # plot train and validation loss
    ax1.plot(data["epoch"], data["train_loss"], label = "Train loss")
    ax1.plot(data["epoch"], data["valid_loss"], label = "Valid loss")

    ax1.axhline(data["valid_loss"].min(),
                linestyle= (0, (5, 10)), linewidth = 0.5)
    ax1.axvline(data["valid_loss"].idxmin(), 
                linestyle = (0, (5, 10)), linewidth = 0.5)
    ax1.text(11, data['valid_loss'].min(), 'min valid loss',
             backgroundcolor='white', va='center', size=7.5)
    
    ax2.plot(data['epoch'], data['train_acc'], label='Train Accuracy')
    ax2.plot(data['epoch'], data['valid_acc'], label='Valid Accuracy')

    ax1.legend()
    ax1.set_title('Running Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Running Accuracy', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(f'runs/logs/{filename}_plot.png')
    plt.show()
    plt.close()

# plot confusion matrices
def plot_confmat(train_mat, test_mat, classes, filename):
    train_mat = pd.DataFrame(train_mat.numpy(), index=classes, columns=classes)
    test_mat = pd.DataFrame(test_mat.numpy(), index=classes, columns=classes)

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(121)
    ax = sns.heatmap(train_mat, annot=True, cmap='tab20c',
                    fmt='d', annot_kws={'size': 18})
    ax.set_title('Confusion Matrix (Train Set)', fontweight='bold')
    ax.set_xlabel('Predicted Classes', fontweight='bold')
    ax.set_ylabel('Actual Classes', fontweight='bold')

    ax = fig.add_subplot(122)
    ax = sns.heatmap(test_mat, annot=True, cmap='tab20c',
                    fmt='d', annot_kws={'size': 18})
    ax.set_title('Confusion Matrix (Test Set)', fontweight='bold')
    ax.set_xlabel('Predicted Classes', fontweight='bold')
    ax.set_ylabel('Actual Classes', fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{filename}') # before ..runs/logs/{filename}
    plt.show()
    plt.close()

# mask generation
def apply_mask(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# plot grad-cam on the image using a given models
def plot_gradcam(image, vgg_cam, res_cam, dense_cam):
    image = unnormaliz_img(image)
    name_dict = {
        'Original Image': image,
        'GradCAM (VGG-16)': apply_mask(image, vgg_cam),
        'GradCAM (ResNet-18)': apply_mask(image, res_cam),
        'GradCAM (DenseNet-121)': apply_mask(image, dense_cam)
    }

    plt.style.use('seaborn-notebook')
    fig = plt.figure(figsize=(20, 4))
    for i, (name, img) in enumerate(name_dict.items()):
        ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
        if i:
            img = img[:, :, ::-1]
        ax.imshow(img)
        ax.set_xlabel(name, fontweight='bold')

    fig.suptitle(
        'Localization with Gradient based Class Activation Maps',
        fontweight='bold', fontsize=16
    )
    plt.tight_layout()
    fig.savefig('outputs/grad_cam.png')
    plt.show()
    plt.close()
    
    