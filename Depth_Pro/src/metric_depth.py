import depth_pro
import numpy as np
import cv2

image_path="C:/Users/jayji/OneDrive/Documents/Monocular Depth Estimation/ml-depth-pro/data/example.jpg"

img = cv2.imread(image_path)
img = cv2.resize(img, (640, 480))
height, width = img.shape[:2]

depth_model,transform=depth_pro.create_model_and_transforms()
depth_model.eval()

image, _, f_px = depth_pro.load_rgb(image_path)
depth_input=transform(image)

prediction=depth_model.infer(depth_input, f_px=f_px)
depth=prediction["depth"]

depth_np = depth.squeeze().cpu().numpy()

depth_value = depth_np[height // 2, width // 2]

text = f"Depth: {depth_value:.2f} meters"
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Depth Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

depth_np_normalized = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
inv_depth_np_normalized = 1 - depth_np_normalized
depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
cv2.imshow('Inverted Depth Map', depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()