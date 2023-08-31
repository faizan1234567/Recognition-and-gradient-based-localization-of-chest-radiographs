"""
Utils functions
---------------

chest x ray recognition
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import sklearn

import torch

# device type
device = "cuda" if torch.cuda.is_available() else "cpu"

# get total number of predictions
def get_all_preds(model, loader):
    model.eval()
    with torch.no_grad():
        all_preds = torch.tensor([], device=device)
        for batch in loader:
            images = batch[0].to(device)
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds 