import numpy as np
import cv2  # OpenCV for image processing
import os
from PIL import Image  # For loading and manipulating images
from ultralytics import YOLO

# Function to calculate the angle between three points (P1, P2, P3)
def calculate_angle(P1, P2, P3):
    """Calculate the angle between three points (P1, P2, P3) using cosine rule."""
    v1 = P1 - P2
    v2 = P3 - P2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Detailed classifier for human actions
def classify_action(keypoints):
    """Classify the action of the person based on keypoints, including running and walking."""
    
    # Extract keypoints for hips, knees, ankles, and shoulders
    left_hip = keypoints[11]  # (x, y) for left hip
    right_hip = keypoints[12]  # (x, y) for right hip
    left_knee = keypoints[13]  # (x, y) for left knee
    right_knee = keypoints[14]  # (x, y) for right knee
    left_ankle = keypoints[15]  # (x, y) for left ankle
    right_ankle = keypoints[16]  # (x, y) for right ankle
    nose = keypoints[0]  # (x, y) for nose (used for height)
    left_shoulder = keypoints[5]  # (x, y) for left shoulder
    right_shoulder = keypoints[6]  # (x, y) for right shoulder

    # Calculate knee angles (hip, knee, ankle)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    # Calculate the body height (nose to ankle distance)
    ankle_center = np.mean([left_ankle, right_ankle], axis=0)
    body_height = np.linalg.norm(nose - ankle_center)
    
    # Calculate hip angles (shoulder, hip, knee)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    # Calculate the vertical spread (nose to ankle y-difference)
    vertical_spread = abs(nose[1] - ankle_center[1])  # Only Y-coordinate difference

    # Calculate stride length (distance between left and right ankles)
    stride_length = np.linalg.norm(left_ankle - right_ankle)
    
    # Adjust thresholds
    standing_knee_angle_threshold = 150  # Adjusted knee angle threshold for standing
    sitting_knee_angle_threshold = 100   # Knee angle threshold for sitting
    standing_body_height_threshold = 100  # Adjusted body height threshold for standing
    lying_body_height_threshold = 150     # Adjusted body height threshold for lying down
    sitting_hip_angle_threshold = 120    # Hip angle threshold for sitting
    standing_hip_angle_threshold = 165   # Hip angle threshold for standing (close to 180)
    
    walking_stride_threshold = 60  # Minimum stride length for walking (adjust based on data)
    running_stride_threshold = 120  # Minimum stride length for running (adjust based on data)
    
    # Check if vertical spread is low, classify as lying down
    lying_down_vertical_threshold = 100  # You can tune this based on experiments
    
    if vertical_spread < lying_down_vertical_threshold:
        return "Lying Down"

    # Check for standing based on knee and hip angles
    if (left_knee_angle > standing_knee_angle_threshold and right_knee_angle > standing_knee_angle_threshold and
        left_hip_angle > standing_hip_angle_threshold and right_hip_angle > standing_hip_angle_threshold):
        if body_height > standing_body_height_threshold:
            return "Standing"
        elif body_height < lying_body_height_threshold:
            return "Lying Down"

    # Check for sitting based on knee and hip angles
    elif (left_knee_angle < sitting_knee_angle_threshold or right_knee_angle < sitting_knee_angle_threshold) or \
         (left_hip_angle < sitting_hip_angle_threshold or right_hip_angle < sitting_hip_angle_threshold):
        return "Sitting"

    # Check for walking based on stride length and moderate knee angles
    elif stride_length > walking_stride_threshold and stride_length < running_stride_threshold:
        return "Walking"
    
    # Check for running based on stride length and dynamic knee angles
    elif stride_length > running_stride_threshold:
        return "Running"
    
    return "Unknown"





# Function to load the appropriate icon based on the action or object class
def load_icon(icon_name, icons_folder):
    """Loads the icon and returns it as a numpy array."""
    icon_path = os.path.join(icons_folder, f"{icon_name}.png")
    if os.path.exists(icon_path):
        icon = Image.open(icon_path).convert('RGBA')  # Open the icon image with transparency
        return np.array(icon)
    else:
        print(f"Icon for {icon_name} not found in {icons_folder}.")
        return None

# Function to check if two bounding boxes overlap
def check_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    overlap = not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)
    return overlap

# Function to move bounding boxes apart with smarter horizontal or vertical separation
def move_bounding_boxes(bbox1, bbox2, shift_amount=20):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate the center points of the bounding boxes
    center_x1 = (x1_min + x1_max) / 2
    center_y1 = (y1_min + y1_max) / 2
    center_x2 = (x2_min + x2_max) / 2
    center_y2 = (y2_min + y2_max) / 2
    
    # Decide whether to shift horizontally or vertically based on which axis has more overlap
    horizontal_overlap = abs(center_x1 - center_x2) < (x1_max - x1_min + x2_max - x2_min) / 2
    vertical_overlap = abs(center_y1 - center_y2) < (y1_max - y1_min + y2_max - y2_min) / 2
    
    # Move horizontally if horizontal overlap exists
    if horizontal_overlap:
        if center_x1 < center_x2:
            x1_min -= shift_amount
            x1_max -= shift_amount
        else:
            x1_min += shift_amount
            x1_max += shift_amount
    
    # Move vertically if vertical overlap exists
    if vertical_overlap:
        if center_y1 < center_y2:
            y1_min -= shift_amount
            y1_max -= shift_amount
        else:
            y1_min += shift_amount
            y1_max += shift_amount

    return [x1_min, y1_min, x1_max, y1_max]

# Function to resolve overlaps between bounding boxes
def resolve_overlap(bbox_list, shift_amount=20):
    """Resolve overlaps between bounding boxes by shifting them until no overlap exists."""
    for i in range(len(bbox_list)):
        for j in range(i + 1, len(bbox_list)):
            # Continue adjusting until no overlap exists between bbox i and bbox j
            while check_overlap(bbox_list[i], bbox_list[j]):
                bbox_list[i] = move_bounding_boxes(bbox_list[i], bbox_list[j], shift_amount)
    return bbox_list

# Function to overlay icons based on the position (top-left or top-right)
# Function to overlay icons based on the position (top-left or top-right) with min/max icon size constraints
# Function to overlay icons based on the position (top-left or top-right) with min/max icon size constraints
# Sharpening function
def apply_sharpening(image):
    """Apply a sharpening filter to enhance the sharpness of the image."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Updated overlay_icon function with sharpening
def overlay_icon(white_background, icon, bbox, position="left", min_icon_size=150, max_icon_size=200, shift_amount=10):
    """Overlay a resized and sharpened icon within the white background, constrained by min and max icon size."""
    x1, y1, x2, y2 = map(int, bbox)

    # Determine where to place the icon (top-left or top-right)
    icon_h, icon_w, _ = icon.shape

    # Calculate the bounding box dimensions
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Set the icon size based on the bounding box size, constrained by min and max size
    icon_size = min(max_icon_size, max(min_icon_size, max(bbox_width, bbox_height)))

    # Maintain the aspect ratio of the icon
    aspect_ratio = icon_w / icon_h
    if icon_w > icon_h:
        new_w = icon_size
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = icon_size
        new_w = int(new_h * aspect_ratio)

    # Resize the icon with high-quality interpolation
    icon_resized = cv2.resize(icon, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Apply sharpening to the resized icon
    icon_resized = apply_sharpening(icon_resized)

    # Calculate the position to overlay the icon (either top-left or top-right)
    if position == "right":
        x1_icon = x2 - new_w  # Place it at the top-right
        y1_icon = y1 - shift_amount  # Move the icon above the bounding box slightly
    else:
        x1_icon = x1 - new_w - shift_amount  # Place it to the left of the bounding box
        y1_icon = y1 - shift_amount  # Move the icon slightly above the top of the bounding box

    # Ensure the icon stays within the bounds of the image
    x1_icon = max(0, x1_icon)
    y1_icon = max(0, y1_icon)
    
    x2_icon = min(white_background.shape[1], x1_icon + new_w)
    y2_icon = min(white_background.shape[0], y1_icon + new_h)

    # Adjust icon dimensions if they exceed the white background
    icon_resized = cv2.resize(icon_resized, (x2_icon - x1_icon, y2_icon - y1_icon))

    # Overlay the icon on the white background with boundary checks
    icon_rgba = cv2.cvtColor(icon_resized, cv2.COLOR_RGBA2BGRA)
    alpha_s = icon_rgba[:, :, 3] / 255.0  # Extract the alpha channel for transparency
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):  # Overlay only the RGB channels
        white_background[y1_icon:y2_icon, x1_icon:x2_icon, c] = (
            alpha_s * icon_rgba[:y2_icon - y1_icon, :x2_icon - x1_icon, c] + 
            alpha_l * white_background[y1_icon:y2_icon, x1_icon:x2_icon, c]
        )




# Increase icon size for detected objects and persons
scaling_factor = 3  # Set this value to increase icon size (e.g., 1.5 for 50% bigger icons)

# Load YOLOv8 models for pose and object detection
pose_model = YOLO("yolov8n-pose.pt")
detection_model = YOLO("yolov8n.pt")

# Load the image and run detection
image_path = 'c:/Users/mugun/Downloads/test20.jpg'
image = cv2.imread(image_path)

pose_results = pose_model(image_path)
detection_results = detection_model(image_path)

# Create a white background of the same size as the original image
image_shape = image.shape
white_background = np.ones(image_shape, dtype=np.uint8) * 255  # White background

# Icons directory (ensure your icons are in this directory)
icons_folder = 'C:/Users/mugun/Downloads/test'

# List to store bounding boxes and icons
object_bboxes = []
object_icons_to_display = []

# Bounding box for the person detected by object detection
person_bbox = None

# Process object detection results and retrieve the bounding box for the person
confidence_threshold = 0.5
for result in detection_results:
    boxes = result.boxes
    classes = result.names
    
    for i, box in enumerate(boxes):
        confidence = box.conf.item()
        class_id = int(box.cls.item())
        class_name = classes[class_id]
        
        # If the detected class is "person", we capture the person's bounding box
        if confidence > confidence_threshold:
            if class_name.lower() == "person":
                person_bbox = box.xyxy.cpu().numpy()[0]
            else:
                # Load icons for other detected objects
                object_icon = load_icon(class_name.lower(), icons_folder)
                if object_icon is not None:
                    object_icons_to_display.append(object_icon)
                    object_bboxes.append(box.xyxy.cpu().numpy()[0])

# If we have detected a person, we will classify their action using pose detection
min_icon_size = 120  # Minimum icon size
max_icon_size = 190  # Maximum icon size

# If we have detected a person, classify their action and overlay the action icon
if person_bbox is not None:
    print("Person detected, processing action...")

    for result in pose_results:
        keypoints = result.keypoints.xy.cpu().numpy().squeeze(0)
        action = classify_action(keypoints)
        print(f"Detected action: {action}")

        # Load the action icon
        action_icon = load_icon(action.lower(), icons_folder)

        # Check for overlap between action and object bounding boxes
        has_overlap = False
        for obj_bbox in object_bboxes:
            if check_overlap(person_bbox, obj_bbox):
                has_overlap = True
                break

        # If overlap happens, move the action icon to the top-right of the bounding box
        if action_icon is not None:
            x1, y1, x2, y2 = map(int, person_bbox)
            obj_width = max(20, min(int(x2 - x1), 50))
            obj_height = max(20, min(int(y2 - y1), 50))

            resized_action_icon = Image.fromarray(action_icon).resize((obj_width, obj_height))
            resized_action_icon = np.array(resized_action_icon)

            # Overlay the action icon either on the right or left depending on overlap
            position = "right" if has_overlap else "left"
            overlay_icon(white_background, resized_action_icon, person_bbox, position=position, min_icon_size=min_icon_size, max_icon_size=max_icon_size)

# Resolve overlapping bounding boxes
object_bboxes = resolve_overlap(object_bboxes)

# Process the detected object icons with min and max icon size constraints
for bbox, icon in zip(object_bboxes, object_icons_to_display):
    # Resize icons to fit within their bounding boxes, and constrain the size
    x1, y1, x2, y2 = map(int, bbox)
    obj_width = max(20, min(int(x2 - x1), 50))
    obj_height = max(20, min(int(y2 - y1), 50))

    resized_icon = Image.fromarray(icon).resize((obj_width, obj_height))
    resized_icon = np.array(resized_icon)

    # Overlay the resized object icon with min/max size constraints
    overlay_icon(white_background, resized_icon, bbox, position="left", min_icon_size=min_icon_size, max_icon_size=max_icon_size)

# Save the final image with icons
output_path = 'c:/Users/mugun/Downloads/detection_with_actions_output_no_overlap.png'
cv2.imwrite(output_path, white_background)

print(f"Output saved to {output_path}")