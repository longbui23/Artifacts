import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import time

def data_visualizer(image_path, label_path):
    """
    Visualize the image in training/testing dataset with label and simulate AR effects.
    Enable clicking on bounding boxes or polygons to zoom into the area.
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
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Color cycling for rectangles and polygons
    bounding_boxes = []
    polygons = []
    start_time = time.time()

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

                bounding_boxes.append((x1, y1, x2, y2, class_id))

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

                polygons.append((np.array(points, dtype=np.int32), class_id))

        except Exception as e:
            print(f"Error processing line: {line}, Error: {e}")

    # Define mouse callback function
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked inside any rectangle
            for box in bounding_boxes:
                x1, y1, x2, y2, class_id = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Zoom into the clicked area
                    zoomed_area = image[y1:y2, x1:x2]
                    zoomed_area = cv2.resize(zoomed_area, (640, 640))
                    cv2.imshow("Zoomed Area", zoomed_area)

            # Check if clicked inside any polygon
            for poly, class_id in polygons:
                if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                    # Zoom into the bounding box of the polygon
                    x1, y1, w, h = cv2.boundingRect(poly)
                    zoomed_area = image[y1:y1 + h, x1:x1 + w]
                    zoomed_area = cv2.resize(zoomed_area, (640, 640))
                    cv2.imshow("Zoomed Area", zoomed_area)

    cv2.namedWindow("Interactive AR Visualization")
    cv2.setMouseCallback("Interactive AR Visualization", on_mouse)

    while True:
        # Copy the image to draw AR effects
        frame = image.copy()
        elapsed_time = time.time() - start_time

        for i, (x1, y1, x2, y2, class_id) in enumerate(bounding_boxes):
            # Dynamic color cycling for rectangles
            color = colors[int(elapsed_time) % len(colors)]

            # "Shiny" effect by alternating thickness
            thickness = 2 + (i % 2)

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"Class: {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for i, (poly, class_id) in enumerate(polygons):
            # Dynamic color cycling for polygons
            color = colors[int(elapsed_time) % len(colors)]

            # "Shiny" effect by alternating thickness
            thickness = 2 + (i % 2)

            # Draw polygon and label
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=thickness)
            x, y = poly[0] # Use the first point as label position
            cv2.putText(frame, f"Class: {class_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the result
        cv2.imshow("Interactive AR Visualization", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def predict(file_path):
    """
    Detect objects in the image using YOLOv5 model.
    :param file_path: Path to the image.
    """
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

    # Inference
    results = model(file_path)

    # Results
    results.print()
    results.show()

# Example usage
data_visualizer('test_1000_jpg.rf.5227b83f3fb933d428c02212cb6d72df.jpg', 'test_1000_jpg.rf.5227b83f3fb933d428c02212cb6d72df.txt')