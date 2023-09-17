# Recognition and Gradient-based Localization of Chest Radiographs

## Contents
- Introduction
- Overview
- Pipeline
- Results
- Installation
- Usage
- Todo
- Conclusion
- Acknowlegdments

## Introduction
Chest diseases such as COVID-19, Pneumonia, and other abnormalities are among ubiquities medical conditions in the world. They are usually done using pathological photographs of a patient’s lungs. There are a lot of details and essential clues, but manual evaluation may not be as fast and accurate. Therefore, it’s important to use effective and efficient diagnoses with minimal cost, time, and high accuracy. [1]  Furthermore, diagnoses become even harder if there is a change on chest x-ray images related to Pneumonia manifestation or other patient’s medical history. Moreover, patients might have other pre-existing conditions such as bleeding, pulmonary edema, lung cancer, atelectasis, or surgical reasons. The goal of using AI to highlight specific regions where pneumonia or a disease area exists. [2] This project aims to train state-of-the-art deep neural networks on large scale chest x-ray database to improve the quality of diagnosis of three diseases categories such as ```COVID-19```, ```Pneumonia```, and ```lung opacity```. In addition, we use ```normal``` class to differentiate between patients and normal people. Initially, we are training on ```ResNet18``` [3], ```VGG16``` [4], and ```DenseNet121``` [5]. However, later we will design a state-of-the-art model based on observations and experiments using attention models such ```vision transformers```. [6]

## Overview
This repository uses chest radiograph dataset from Kaggle [7], [8]. It has a total of ```21165``` examples of chest x-ray categorized under ```COVID-19```, ```Pneumonia```, ```Lung Opacity```, and ```Normal```.  Furthermore, some preprocessing transforms have been defined. To get the insight from the data, we used image understanding models such as ```ResNet18``` [3], ```DenseNet121``` [5], and ```VGG16``` [4] trained on ```ImageNet``` [9] Dataset, however, we fine-tunned it on the chest radiographs dataset.   The results will be presented in a section later. Finally, by using Gradient weighted class activation maps (```Grad-CAM```) [10], models high confidence regions have been localized.

## Project Pipeline
1. [Dataset Exploration](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
2. [Dataset Information](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
   |Type|COVID-19|Lung Opacity|Normal|Viral Pneumonia|Total|
   |:-|-:|-:|-:|-:|-:|
   |Train|3496|5892|10072|1225|20685|
   |Val|60|60|60|60|240|
   |Test|60|60|60|60|240|
3. [Fine-tune ResNet, VGG16, and DenseNet121](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/pretrained_models.py)
  1. [Dataset Transformations](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/dataset/data.py#L25)
  2. [Handling imbalanced dataset](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/dataset/data.py#L96)
  3. [Loading prepretrained models](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/pretrained_models.py#L34)
  4. [Hyperparameters used](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/configs/configs.yaml)
       - |Hyper-parameters||
         |:-|-:|
         |Learning rate|`3e-5`|
         |Batch Size|`64`|
         |Number of Epochs|`10`|
       - |Loss Function|Optimizer|
         |:-:|:-:|
         |`Categorical Cross Entropy`|`Adam`|
 5. [Loading dataset](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/dataset/data.py)
 6. [Training](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/train.py)
 7. [Inference](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/test.py)
 8. [Gradient-based Localization](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/draw_cam.py)

 ## Results
 1. [Plotting running losses and accuracies](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/utils.py#L97)
 - |Model|Loss and Accuracy Plots|
   |:-:|:-:|
   |VGG-16|![vgg_plot](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/runs/logs/vgg16_plot.png)|
   |ResNet-18|![res_plot](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/runs/logs/resnet18_plot.png)|
   |DenseNet-121|![dense_plot](https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs/blob/master/runs/logs/densenet121_plot.png)|



  <table>
  <tr>
  <th></th>
  <th>VGG-16</th>
  <th>ResNet-18</th>
  <th>DenseNet-121</th>
  </tr>
  <tr>
  <td>

  |__Pathology__|
  |:-|
  |COVID-19|
  |Lung Opacity|
  |Normal|
  |Viral Pneumonia|

  </td>
  <td>

  |Accuracy|Precision|Recall|F1-Score|
  |-:|-:|-:|-:|
  |0.978 |0.983 |0.936 |0.959 |
  |0.953 |0.85  |0.962 |0.902 |
  |0.953 |0.933 |0.888 |0.910 |
  |0.995 |1.0   |0.983 |0.991 |
              
  </td>
  <td>

  |Accuracy|Precision|Recall|F1-Score|
  |-:|-:|-:|-:|
  |0.9871|0.9667|0.9830|0.9748|
  |0.9664|0.8667|1.0000|0.9286|
  |0.9664|1.0000|0.8823|0.9375|
  |0.9957|1.0000|0.9836|0.9917|
              
  </td>
  <td>

  |Accuracy|Precision|Recall|F1-Score|
  |-:|-:|-:|-:|
  |0.9957|0.9833|1.0000|0.9916|
  |0.9623|0.9167|0.9322|0.9244|
  |0.9623|0.9500|0.9047|0.9268|
  |0.9957|0.9833|1.0000|0.9916|
              
  </td>
  </tr>
  <tr> 


 ## Installation
 ```bash
 git clone https://github.com/faizan1234567/Recognition-and-gradient-based-localization-of-chest-radiographs.git
 cd Recognition-and-gradient-based-localization-of-chest-radiographs
```
Create and activate Anaconda Environment
```bash
conda create -n chest-xray python=3.9.0
conda activate chest-xray
```
Now install all the required dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Installation Complete !

## Usage
To get play with data loading, run the following script
```python
python dataset/data.py 
```
To train on your dataset
```python
python train.py -h
python train.py --epochs 100 --learning_rate 3e-5 --batch 32 --save runs/ --workers 8 --model 'resnet18' 
```
To run inference on test dataset
```python
python test.py -h
python test.py --batch 32 --weights <path> --model 'resnet18' --classes 4 --kind 'test'
--subset
```
To run Grad-CAM for localizating activations
```python
python draw_cam.py -h
python draw_cam.py --model 'resnet18' --output <path> --connfig configs/configs.yaml --save <path>
```
If you face any issue in installation and usage, please create an issue. If you have any ideas for improvments kindly create a PR.

## TODO
 - Automatic hyperparameters optimization
 - Multi-GPU training support
 - Other Gradient based localization techniques integration
 - Other state-of-the-art models architectures design 
 - Adding Hydra configurations
 - Adding Data Version Control


## Conclusion
In this repository, three image understanding models namely ```DenseNet121```, ```ResNet18```, and ```VGG16``` have fine-tuned on the x-ray dataset. Since the dataset is pretty unbalanced, oversampling stretegy helped with imbalanced dataset. In addition to that, ```Grad-CAM``` localization have increased model's interpretablility and chances for improvments. The models have been trained on ```10 epochs``` which are not enough. Based on the results obtained, more data augmentation, better hyperparameters optimization, and model architecutre should be designed for good accuracy. 



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

[11] https://github.com/priyavrat-misra/xrays-and-gradcam

