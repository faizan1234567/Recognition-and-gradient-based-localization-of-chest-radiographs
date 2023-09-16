# Recognition and Gradient-based Localization of Chest Radiographs

## Contents
- Introduction
- Overview
- Pipeline
- Results
- Installation
- Usage
- Conclusion
- Todo
- Acknowlegdments

## Introduction
Chest diseases such as COVID-19, Pneumonia, and other abnormalities are among ubiquities medical conditions in the world. They are usually done using pathological photographs of a patient’s lungs. There are a lot of details and essential clues, but manual evaluation may not be as fast and accurate. Therefore, it’s important to use effective and efficient diagnoses with minimal cost, time, and high accuracy. [1]  Furthermore, diagnoses become even harder if there is a change on chest x-ray images related to Pneumonia manifestation or other patient’s medical history. Moreover, patients might have other pre-existing conditions such as bleeding, pulmonary edema, lung cancer, atelectasis, or surgical reasons. The goal of using AI to highlight specific regions where pneumonia or a disease area exists. [2] This project aims to train state-of-the-art deep neural networks on large scale chest x-ray database to improve the quality of diagnosis of three diseases categories such as ```COVID-19```, ```Pneumonia```, and ```lung opacity```. In addition, we use ```normal``` class to differentiate between patients and normal people. Initially, we are training on ```ResNet18``` [3], ```VGG16``` [4], and ```DenseNet121``` [5]. However, later we will design a state-of-the-art model based on observations and experiments using attention models such ```vision transformers```. [6]

## Project Pipeline
1. [Dataset Exploration]()
2. Dataset Information
   |Type|COVID-19|Lung Opacity|Normal|Viral Pneumonia|Total|
   |:-|-:|-:|-:|-:|-:|
   |Train|3496|5892|10072|1225|20685|
   |Val|60|60|60|60|240|
   |Test|60|60|60|60|240|
3. [Fine-tune ResNet, VGG16, and DenseNet121](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/pretrained_models.py)
  1. [Dataset Transformations](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/dataset/data.py#L25)
  2. [Handling imbalanced dataset](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/dataset/data.py#L96)
  3. [Loading prepretrained models](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/pretrained_models.py#L34)
  4. Hyperparameters used
       - |Hyper-parameters||
         |:-|-:|
         |Learning rate|`0.00003`|
         |Batch Size|`32`|
         |Number of Epochs|`25`|
       - |Loss Function|Optimizer|
         |:-:|:-:|
         |`Categorical Cross Entropy`|`Adam`|
## Overview
This repository uses chest radiograph dataset from Kaggle [7], [8]. It has a total of ```21165``` examples of chest x-ray categorized under ```COVID-19```, ```Pneumonia```, ```Lung Opacity```, and ```Normal```.  Furthermore, some preprocessing transforms have been defined. To get the insight from the data, we used image understanding models such as ```ResNet18``` [3], ```DenseNet121``` [5], and ```VGG16``` [4] trained on ```ImageNet``` [9] Dataset, however, we fine-tunned it on the chest radiographs dataset.   The results will be presented in a section later. Finally, by using Gradient weighted class activation maps (```Grad-CAM```) [10], models high confidence regions have been localized.

## Acknowledgements
[1]. 	H. Su et al., “Multilevel threshold image segmentation for COVID-19 chest radiography: A framework using horizontal and vertical multiverse optimization,” Comput. Biol. Med., vol. 146, no. May, p. 105618, 2022, doi: 10.1016/j.compbiomed.2022.105618.

[2]	I. Sirazitdinov, M. Kholiavchenko, T. Mustafaev, Y. Yixuan, R. Kuleev, and B. Ibragimov, “Deep neural network ensemble for pneumonia localization from a large-scale chest x-ray database,” Comput. Electr. Eng., vol. 78, pp. 388–399, 2019, doi: 10.1016/j.compeleceng.2019.08.004.

[3]	K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” Dec. 2015, [Online]. Available: http://arxiv.org/abs/1512.03385

[4]	“ K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.”.

[5]	G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” Proc. - 30th IEEE Conf. Comput. Vis. Pattern Recognition, CVPR 2017, vol. 2017-January, pp. 2261–2269, 2017, doi: 10.1109/CVPR.2017.243.

[6]	A. Vaswani, “Attention Is All You Need,” no. Nips, 2017.

[7]	M. E. H. Chowdhury et al., “Can AI Help in Screening Viral and COVID-19 Pneumonia?,” IEEE Access, vol. 8, pp. 132665–132676, 2020, doi: 10.1109/ACCESS.2020.3010287.

[8]	T. Rahman et al., “Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images,” Comput. Biol. Med., vol. 132, no. March, p. 104319, 2021, doi: 10.1016/j.compbiomed.2021.104319.

[9]	J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255. doi: 10.1109/CVPR.2009.5206848.

[10]	R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” Int. J. Comput. Vis., vol. 128, no. 2, pp. 336–359, 2020, doi: 10.1007/s11263-019-01228-7.

