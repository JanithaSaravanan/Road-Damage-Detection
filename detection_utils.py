from ultralytics import YOLO
import cv2
import os
import torch
import torchvision.transforms as T
import numpy as np

#  1. Load YOLOv8 and MiDaS models 
model = YOLO("project_files/bestfinal.pt")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# SEVERITY CALCULATION 

def calculate_severity_style(relative_area, normalized_depth_inverse):
   
    # Severity score 
    severity_score = relative_area * normalized_depth_inverse

    # Determine severity level and styling (BGR format for OpenCV)
    if severity_score >= 0.6:
        color_bgr = (0, 0, 255)  # RED
        label = "DANGER (HIGH)"
    elif severity_score >= 0.3:
        color_bgr = (0, 255, 255)  # YELLOW
        label = "WARNING (MEDIUM)"
    else:
        color_bgr = (0, 255, 0)  # GREEN
        label = "SAFE (LOW)"

    return {
        "score": round(severity_score, 4),
        "color_bgr": color_bgr,
        "label": label
    }

# IMAGE PREPROCESSING UTILITY 

def unsharp_mask(image, sigma=1.0, strength=1.5):
  
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to create the "unsharp" mask
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Calculate the sharpened image
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    
    # Convert back to BGR to match original image format
    sharpened_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return sharpened_bgr

# MAIN PROCESSING FUNCTION 

def process_media(input_path):
    
    #  Fix 1: Normalize path for Windows and remove extra quotes/spaces
    input_path = input_path.strip().replace("\\", "/")
    if input_path.startswith('"') and input_path.endswith('"'):
        input_path = input_path[1:-1]

    #  Fix 2: Check if file exists
    if not os.path.exists(input_path):
        print(f" File not found at: {input_path}")
        return None
    else:
        print(f" Processing file: {input_path}")

    # Detect if it's video or image
    is_video = False
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened():
        is_video = True
    else:
        frame = cv2.imread(input_path)
        if frame is None:
            print(" Could not read file as image or video.")
            return None


    # Ensure the output directory exists
    result_path = "static/outputs"
    os.makedirs(result_path, exist_ok=True)
    output_filename = f"processed_{os.path.basename(input_path)}"
    output_filepath = os.path.join(result_path, output_filename)

    if is_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
    else:
        out = None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    last_frame_detections = [] # To store structured data for the Flask app

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        
        # If it's an image, the frame is already loaded, and we break after processing
        
        # Get frame dimensions for relative area calculation
        H, W = frame.shape[:2]
        
        #  PREPROCESSING OF EACH FRAME 
        frame_denoised = cv2.bilateralFilter(frame, 9, 75, 75)
        lab = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        l_clahe_merged = cv2.merge((l_clahe, a, b))
        frame_enhanced = cv2.cvtColor(l_clahe_merged, cv2.COLOR_LAB2BGR)
        frame_preprocessed = unsharp_mask(frame_enhanced)

        #  Depth Estimation 
        img_rgb = cv2.cvtColor(frame_preprocessed, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for severity calculation (0=closest, 1=farthest)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        # Avoid division by zero if depth is constant (e.g., solid color image)
        if depth_max == depth_min:
            normalized_depth_map = np.ones_like(depth_map) * 0.5
        else:
            normalized_depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            
        # Reset detection list for the current frame
        frame_detections = []

      # YOLOv8 Detection 
        results = model(frame_preprocessed, conf=0.5)

        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id].lower()  

        # Bounding box dimensions
        box_w = x2 - x1
        box_h = y2 - y1
        relative_area = (box_w * box_h) / (W * H)

        # Depth regions
        region_depth = normalized_depth_map[y1:y2, x1:x2]
        raw_depth_region = depth_map[y1:y2, x1:x2]

        # Defaults
        distance_in_m, distance_in_cm = 0, 0
        severity_data = {"score": 0, "color_bgr": (255, 255, 255), "label": "info"}

        # Distance
        if raw_depth_region.size > 0:
            k = 200.0  
            distance_in_m = (k / raw_depth_region.mean())
            distance_in_cm = distance_in_m * 100

        # severity 
        if class_name in ["pothole"] and region_depth.size > 0:
            avg_depth = region_depth.mean()
            normalized_depth_inverse = avg_depth
            severity_data = calculate_severity_style(relative_area, normalized_depth_inverse)
            box_color = severity_data["color_bgr"]
            text = f"{class_name} | severity: {severity_data['label'].lower()} | dist: {distance_in_cm:.1f} cm"

        # Speed bump 
        elif class_name == "speed_bump":
            box_color = (255, 0, 0)  # blue
            text = f"{class_name} | dist: {distance_in_cm:.1f} cm"

        # Manhole 
        elif class_name == "manhole":
            box_color = (192, 192, 192)  # gray
            text = f"{class_name} | dist: {distance_in_cm:.1f} cm"

        else:
            box_color = (255, 255, 255)
            text = f"{class_name} | dist: {distance_in_cm:.1f} cm"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Draw black background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(frame,
                      (x1, y1 - text_height - baseline - 10),
                      (x1 + text_width, y1 - 10),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (x1, y1 - baseline - 5),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Sobel edge detection 
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobelx, sobely)
        sobel_mag = cv2.convertScaleAbs(sobel_mag)
        _, thresh = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours, -1, (255, 200, 150), 2)
        frame[y1:y2, x1:x2] = roi

        # Store detection info
        frame_detections.append({
            "id": i + 1,
            "object": class_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 2),
            "distance_m": round(distance_in_m, 2),
            "relative_area": round(relative_area, 4),
            "avg_normalized_depth": round(region_depth.mean(), 4) if region_depth.size > 0 else None,
            "severity": {
                "score": severity_data["score"],
                "label": severity_data["label"].lower() if class_name in ["pothole"] else "info"
            }
        })


        # Save output
        if is_video:
            out.write(frame)
        else:
            cv2.imwrite(output_filepath, frame)
            break # Exit loop after image processing

    if is_video:
        cap.release()
        out.release()
    
    # Return the output path and the detections from the last frame/image
    return {
        "output_path": output_filepath,
        "detections": last_frame_detections,
        "file_path": input_path
    }