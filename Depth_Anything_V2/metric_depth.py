from transformers import pipeline
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device=0)

image_path = 'images/floor_toys_1.jpg'
image_cv2 = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

depth_result = pipe(image_pil)
depth_map = depth_result["depth"]

# Convert PIL Image to NumPy array
depth_array = np.array(depth_map)

height, width = depth_array.shape 
center_x, center_y = width // 2, height // 2

metric_distance = depth_array[center_y, center_x]
print(f"Metric distance at the center ({center_x}, {center_y}): {metric_distance:.2f} meters")

plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

# Depth Map
plt.subplot(1, 2, 2)
plt.imshow(depth_array, cmap="plasma")
plt.colorbar(label="Depth (meters)")
plt.title("Depth Map")
plt.axis("off")

plt.show()
