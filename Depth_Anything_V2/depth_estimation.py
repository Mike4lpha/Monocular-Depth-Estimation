import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitb.pth', map_location='cpu', weights_only=True))
model.eval()

raw_img = cv2.imread('images/floor_toys_1.jpg')
depth = model.infer_image(raw_img) # HxW raw depth map
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('raw_img', raw_img)
cv2.imshow('Depth Image', depth_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()