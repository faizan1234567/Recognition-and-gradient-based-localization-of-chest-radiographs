"""
Implementation of GradCAM with xray images
------------------------------------------

most of the code for the script 
is borrow from:https://github.com/priyavrat-misra/xrays-and-gradcam/blob/master/grad_cam.py
"""

import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, desired_layer):
        self.model = model.eval()
        self.desired_layer = desired_layer
        self.features_map = []
        self.gradients = []
        
        # register forward and backward hook
        desired_layer.register_forward_hook(self.store_feature_maps)
        desired_layer.register_backward_hook(self.store_gradients)

    # method to store feature maps from the desired layer in the forward pass
    def store_feature_maps(self, module, input, output):
        self.features_map.append(output)

    # store gradients 
    def store_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])
    
    # return cam weights
    def get_weights(self, grads):
        return np.mean(grads, axis= (1, 2))


