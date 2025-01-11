from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device=0)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

cv2.namedWindow("Webcam Depth Estimation", cv2.WINDOW_NORMAL)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        depth_result = pipe(frame_pil)
        depth_map = depth_result["depth"]

        depth_array = np.array(depth_map)

        height, width = depth_array.shape
        center_x, center_y = width // 2, height // 2

        metric_distance = depth_array[center_y, center_x]

        cv2.putText(frame, f"Depth at center: {metric_distance:.2f}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        normalized_depth = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_PLASMA)

        combined = cv2.hconcat([frame, depth_colormap])

        cv2.imshow("Webcam Depth Estimation", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

