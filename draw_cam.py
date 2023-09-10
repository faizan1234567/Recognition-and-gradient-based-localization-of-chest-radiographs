"""
Overlay Gradcam predictions on xray images
------------------------------------------

python draw_cam.py -h
---------------------
"""
# packages
import argparse
import os
import sys
from pretrained_models import get_model
from utils import apply_mask
from grad_cam import GradCAM


