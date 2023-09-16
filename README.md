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
Chest diseases such as COVID-19, Pneumonia, and other abnormalities are among ubiquities medical conditions in the world. They are usually done using pathological photographs of a patient’s lungs. There are a lot of details and essential clues, but manual evaluation may not be as fast and accurate. Therefore, it’s important to use effective and efficient diagnoses with minimal cost, time, and high accuracy. [1]  Furthermore, diagnoses become even harder if there is a change on chest x-ray images related to Pneumonia manifestation or other patient’s medical history. Moreover, patients might have other pre-existing conditions such as bleeding, pulmonary edema, lung cancer, atelectasis, or surgical reasons. The goal of using AI to highlight specific regions where pneumonia or a disease area exists. [2] This project aims to train state-of-the-art deep neural networks on large scale chest x-ray database to improve the quality of diagnosis of three diseases categories such as COVID-19, Pneumonia, and lung opacity. In addition, we use normal class to differentiate between patients and normal people. Initially, we are training on ResNet [3], VGG16 [4], and DenseNet121 [5]. However, later we will design a state-of-the-art model based on observations and experiments. 

## Acknowledgements
[1]. 	H. Su et al., “Multilevel threshold image segmentation for COVID-19 chest radiography: A framework using horizontal and vertical multiverse optimization,” Comput. Biol. Med., vol. 146, no. May, p. 105618, 2022, doi: 10.1016/j.compbiomed.2022.105618.
[2]	I. Sirazitdinov, M. Kholiavchenko, T. Mustafaev, Y. Yixuan, R. Kuleev, and B. Ibragimov, “Deep neural network ensemble for pneumonia localization from a large-scale chest x-ray database,” Comput. Electr. Eng., vol. 78, pp. 388–399, 2019, doi: 10.1016/j.compeleceng.2019.08.004.
[3]	K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” Dec. 2015, [Online]. Available: http://arxiv.org/abs/1512.03385
[4]	“ K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.”.
[5]	G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” Proc. - 30th IEEE Conf. Comput. Vis. Pattern Recognition, CVPR 2017, vol. 2017-January, pp. 2261–2269, 2017, doi: 10.1109/CVPR.2017.243.
