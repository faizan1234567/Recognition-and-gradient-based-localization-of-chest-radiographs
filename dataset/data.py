"""
Load the dataset from the disk
"""
import torch
import torchvision
from torchvision import transforms, datasets
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
def get_transforms(config_file: str = configs_file, 
                   type: str = "train"):
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
    with open(configs_file, 'r') as f:
        config = yaml.safe_load(f)
    if type == "train":
        data_transforms = transforms.Compose([
            transforms.Resize(config["Augmentation"]["resize"]),
            transforms.RandomCrop(config["Augmentation"]["random_crop"]),
            transforms. RandomAffine(config["Augmentation"]["random_affine"]["rotation"],
                                    config["Augmentation"]["random_affine"]["translation"],
                                    config["Augmentation"]["random_affine"]["scaling"],
                                    config["Augmentation"]["random_affine"]["shear"]
            ),
            transforms.RandomHorizontalFlip(config["Augmentation"]["horizontal_flip"]),
            transforms.RandomVerticalFlip(config["Augmentation"]["vertical_flip"]),
            transforms.ToTensor(),
            transforms.Normalize(mean= config["Augmentation"]["mean"],
                                std=  config["Augmentation"]["std"]),
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(config["Augmentation"]["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(mean= config["Augmentation"]["mean"],
                                std=  config["Augmentation"]["std"]),
        ])
    return data_transforms

# code: https://discuss.pytorch.org/t/dataloader-error-trying-to-resize-storage-that-is-not-resizable/177584
def collate_fn(batch):
  return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
}

def load_dataset(config_file: str = configs_file,
                 type: str = "train"):
    """
    Load the dataset from the computer in batches, if needed shuffle the
    dataset

    Parameters
    ----------
    config_file: str
    data_transforms: torchvision.transforms.Compose

    Return
    ------
    data_loader: torch.utils.data.DataLoader
    """
    data_transforms = get_transforms(configs_file, type= type)
    with open(configs_file, 'r') as f:
        config = yaml.safe_load(f)

    xray_dataset = datasets.ImageFolder(root=config["general_configs"]["dataset path"] + "/" + type,
                                                transform=data_transforms)
    
    dataset_loader = torch.utils.data.DataLoader(xray_dataset,
                                             batch_size=config["DataLoader"]["batch_size"], 
                                             shuffle=config["DataLoader"]["data_shuffle"],
                                             num_workers=config["DataLoader"]["num_workers"],
                                             pin_memory = True, drop_last = True,
                                             )
    
    return dataset_loader


if __name__ == "__main__":
    dataset = load_dataset(config_file= configs_file, type = 'val')
    image, labels = next(iter(dataset))
    print(image.shape, labels.shape)
    print(labels)


