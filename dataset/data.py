"""
Load the dataset from the disk
"""
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import WeightedRandomSampler, Subset
import yaml
import sys
import os
from pathlib import Path
import argparse

#add path to sys.path
file = Path(__file__).resolve()
ROOT = file.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT)) # add path
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

configs_file = "configs/configs.yaml"

# some common data transforms
def get_transforms(cfg: dict = {}, 
                   kind: str = "train"):
    """
    Add Image transfroms and data augmentation for making dataset 
    diverse and robusts to realistic changes

    Parameters
    ----------
    config_file: str
    type: str

    Return
    ------
    data_transforms: torchvision.transforms.Compose
    """
    if kind == "train":
        data_transforms = transforms.Compose([
            transforms.Resize(cfg["Augmentation"]["resize"]),
            transforms.RandomHorizontalFlip(cfg["Augmentation"]["horizontal_flip"]),
            transforms.ToTensor(),
            transforms.Normalize(mean= cfg["Augmentation"]["mean"],
                                std=  cfg["Augmentation"]["std"]),
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(cfg["Augmentation"]["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(mean= cfg["Augmentation"]["mean"],
                                std=  cfg["Augmentation"]["std"]),
        ])
    return data_transforms

# code: https://discuss.pytorch.org/t/dataloader-error-trying-to-resize-storage-that-is-not-resizable/177584
def collate_fn(batch):
  return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
}

def load_dataset(config_file = configs_file,
                 batch_size = None,
                 kind: str = "train",
                 subset: bool = False):
    """
    Load the dataset from the computer in batches, if needed shuffle the
    dataset

    Parameters
    ----------
    config_file: str
    data_transforms: torchvision.transforms.Compose
    subset: bool

    Return
    ------
    data_loader: torch.utils.data.DataLoader
    """
    if isinstance(config_file, str):
        with open(configs_file, 'r') as f:
            config = yaml.safe_load(f)
    elif isinstance(config_file, dict):
        config = config_file
        
    data_transforms = get_transforms(config, kind= kind)


    xray_dataset = datasets.ImageFolder(root=config["general_configs"]["dataset splitted"] + "/" + kind,
                                                transform=data_transforms)
    # to handle class unbalanced
    class_freq = torch.as_tensor(xray_dataset.targets).bincount()
    weight = 1 / class_freq
    samples_weight = weight[xray_dataset.targets]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    if subset:
        subset_size = 1000 if kind == 'train' else 100
        subset_indices = torch.randperm(len(xray_dataset))[:subset_size]
        xray_dataset = Subset(xray_dataset, subset_indices)

    dataset_loader = torch.utils.data.DataLoader(xray_dataset,
                                             batch_size=config["DataLoader"]["batch_size"] if batch_size == None else batch_size, 
                                             sampler = sampler if kind == "train" and subset == False else None,
                                             num_workers=config["DataLoader"]["num_workers"],
                                             pin_memory = True, drop_last = True,
                                             )
    
    return dataset_loader


if __name__ == "__main__":
    #BUG: fixed 
    dataset = load_dataset(config_file= configs_file, kind = 'train', subset = True)
    print('total images: {}'.format(len(dataset.dataset)))
    image, labels = next(iter(dataset))
    print(image.shape, labels.shape)
    print(labels)


