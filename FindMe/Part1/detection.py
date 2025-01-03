# Shevles image dection using OpenCV and YOLOv5

#import packages
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import time


import cv2
import numpy as np
import time

def data_visualizer(image_path, label_path):
    """
    Visualize the image in training/testing dataset with label and simulate AR effects.
    :param image_path: Path to the image.
    :param label_path: Path to the label file.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_height, image_width = image.shape[:2]

    # Read label
    try:
        with open(label_path, "r") as f:
            lines = f.read().strip().split("\n")
    except FileNotFoundError:
        raise FileNotFoundError(f"Label file not found at {label_path}")

    # Define AR effect parameters
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Color cycling for rectangles
    start_time = time.time()

    # Draw label boxes with AR effects
    for line in lines:
        values = line.split()
        try:
            class_id = int(values[0])

            # Handle rectangle format
            if len(values) == 5:
                # Parse YOLO format
                center_x = float(values[1])
                center_y = float(values[2])
                width = float(values[3])
                height = float(values[4])

                # Convert to pixel coordinates
                x1 = int((center_x - width / 2) * image_width)
                y1 = int((center_y - height / 2) * image_height)
                x2 = int((center_x + width / 2) * image_width)
                y2 = int((center_y + height / 2) * image_height)

                # Clip to image bounds
                x1 = max(0, min(x1, image_width - 1))
                y1 = max(0, min(y1, image_height - 1))
                x2 = max(0, min(x2, image_width - 1))
                y2 = max(0, min(y2, image_height - 1))

                # AR effect: Dynamic color change
                elapsed_time = time.time() - start_time
                color = colors[int(elapsed_time) % len(colors)]

                # Draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"Class: {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Handle polygon format
            elif len(values) > 5:
                points = []
                for i in range(1, len(values), 2):
                    x = float(values[i])
                    y = float(values[i + 1])

                    # Convert to pixel coordinates
                    x_pixel = int(x * image_width)
                    y_pixel = int(y * image_height)
                    points.append((x_pixel, y_pixel))

                points = np.array(points, dtype=np.int32)

                # AR effect: Glowing edges
                cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=3)

        except Exception as e:
            print(f"Error processing line: {line}, Error: {e}")

    # Simulate AR overlay: Add a simple animation (moving circle)
    elapsed_time = int((time.time() - start_time) * 10) % image_width
    cv2.circle(image, (elapsed_time, 50), 20, (0, 255, 255), -1)

    # Display the result
    resized_image = Image.fromarray(cv2.resize(image[:,:,::-1], (640, 640)))
    resized_image.show()

# Function to detect objects in the image
def predict(file_path):

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

    # Inference
    results = model(file_path)

    # Results
    results.print() 
    results.show()  


predict('Data/test/images/test_1000_jpg.rf.5227b83f3fb933d428c02212cb6d72df.jpg')
