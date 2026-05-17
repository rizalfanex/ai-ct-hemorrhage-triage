import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import pydicom
import cv2
import PIL
import albumentations
import timm
import torchmetrics
import monai

print('===== IMPORT CHECK =====')
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('numpy:', np.__version__)
print('pandas:', pd.__version__)
print('sklearn:', sklearn.__version__)
print('matplotlib:', matplotlib.__version__)
print('pydicom:', pydicom.__version__)
print('opencv:', cv2.__version__)
print('PIL:', PIL.__version__)
print('albumentations:', albumentations.__version__)
print('timm:', timm.__version__)
print('torchmetrics:', torchmetrics.__version__)
print('monai:', monai.__version__)

x = torch.randn(1024, 1024).cuda()
y = x @ x
print('GPU matrix test:', y.shape)
print('STATUS: ALL CORE LIBRARIES OK')
