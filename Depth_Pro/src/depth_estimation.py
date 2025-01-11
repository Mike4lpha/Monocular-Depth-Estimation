from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import depth_pro

# Load model and preprocessing transform.
model, transform = depth_pro.create_model_and_transforms()
model.eval()

image_path = "C:/Users/jayji/OneDrive/Documents/Monocular Depth Estimation/ml-depth-pro/data/example.jpg"
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in meters.

# Convert to NumPy array if depth is a PyTorch tensor.
if isinstance(depth, torch.Tensor):
    depth = depth.cpu().numpy()

depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
depth_image = Image.fromarray(depth_normalized.astype(np.uint8))

plt.figure(figsize=(10, 5))
plt.imshow(depth, cmap="viridis")
plt.colorbar(label="Depth (m)")
plt.title("Depth Map")
plt.show()

depth_image.show()

output_path = "C:/Users/jayji/OneDrive/Documents/Monocular Depth Estimation/ml-depth-pro/results/depth_map.png"
depth_image.save(output_path)
print(f"Depth map saved to {output_path}")
