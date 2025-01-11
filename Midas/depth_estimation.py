from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = "images/pic.jpg"
model_path = "dpt_large.pt"

image = Image.open(image_path).convert("RGB")

processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large", ignore_mismatched_sizes=True)

weights = torch.load(model_path, map_location="cpu", weights_only=True)
model.load_state_dict(weights, strict=False)
model.eval()

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")

depth = Image.fromarray(formatted)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Depth Map")
plt.imshow(depth, cmap="inferno")
plt.axis("off")

plt.tight_layout()
plt.show()

