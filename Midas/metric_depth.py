import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image


focal_length = 931.1829833984375
center_x = 320
center_y = 180

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large", ignore_mismatched_sizes=True)

weights = torch.load("dpt_large.pt", map_location=device)
model.load_state_dict(weights, strict=False)
model.to(device)
model.eval()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)

    inputs = processor(images=image_pil, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_pil.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()

    pixel_x = center_x
    pixel_y = center_y

    depth_value = output[pixel_y, pixel_x]
    scaling_factor = 1.0
    real_world_distance = depth_value * scaling_factor

    print(f"Real-world distance at pixel ({pixel_x}, {pixel_y}): {real_world_distance:.2f} meters")

    cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), 2)
    cv2.imshow('Real-time Depth', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

