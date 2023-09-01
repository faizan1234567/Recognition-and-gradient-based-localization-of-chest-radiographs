"""
Implementation of GradCAM with xray images
------------------------------------------

most of the code for the script 
is borrowed from:https://github.com/priyavrat-misra/xrays-and-gradcam/blob/master/grad_cam.py
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
    
    def __call__(self, image, label=None):
        preds = self.model(image)
        self.model.zero_grad()

        if label is None:
            label = preds.argmax(dim=1).item()

        preds[:, label].backward()

        featuremaps = self.featuremaps[-1].cpu().data.numpy()[0, :]
        gradients = self.gradients[-1].cpu().data.numpy()[0, :]

        weights = self.get_cam_weights(gradients)
        cam = np.zeros(featuremaps.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * featuremaps[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, image.shape[-2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return label, cam


