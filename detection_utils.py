# detection_utils.py

from ultralytics import YOLO
import cv2
import time
import geocoder
import os
import torch
import torchvision.transforms as T
import numpy as np

# ==== 1. Load YOLOv8 and MiDaS models ====
model = YOLO("project_files/best.pt")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

def unsharp_mask(image, sigma=1.0, strength=1.5):
    """Applies an unsharp mask to an image to sharpen it."""
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to create the "unsharp" mask
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Calculate the sharpened image
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    
    # Convert back to BGR to match original image format
    sharpened_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return sharpened_bgr

def process_media(input_path):
    """
    Processes a video or image for pothole detection, applies Sobel edge detection
    inside each pothole bounding box (yellow outline), and saves the output.
    Returns the path to the saved output file.
    """
    is_video = False
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened():
        is_video = True
    else:
        frame = cv2.imread(input_path)
        if frame is None:
            return None  # File not found

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

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        
        # === PREPROCESSING OF EACH FRAME ===
        # 1. Noise Reduction
        frame_denoised = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # 2. Contrast Enhancement (CLAHE)
        # CLAHE operates on grayscale, so convert, apply, then convert back.
        lab = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        l_clahe_merged = cv2.merge((l_clahe, a, b))
        frame_enhanced = cv2.cvtColor(l_clahe_merged, cv2.COLOR_LAB2BGR)
        
        # 3. Image Sharpening (Unsharp Masking)
        frame_preprocessed = unsharp_mask(frame_enhanced)

        # ==== Depth Estimation ====
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

        # ==== YOLOv8 Detection ====
        results = model(frame_preprocessed, conf=0.5)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                pothole_region = depth_map[y1:y2, x1:x2]
                if pothole_region.size > 0:
                    average_depth = pothole_region.mean()
                    k = 200
                    distance_in_cm = (k / average_depth) * 100
                    
                    text = f"Pothole: {conf:.2f} | Dist: {distance_in_cm:.0f}cm"

                    # ==== Draw YOLO bounding box ====
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    text_color = (255, 255, 255)  # White
                    text_background_color = (0, 0, 0)  # Black
                    
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                    cv2.rectangle(frame, (x1, y1 - text_height - baseline - 10), 
                                  (x1 + text_width, y1 - 10), text_background_color, -1)
                    cv2.putText(frame, text, (x1, y1 - baseline - 5),
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    # ==== Sobel Edge Detection inside pothole ====
                    roi = frame[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
                    sobel_mag = cv2.magnitude(sobelx, sobely)
                    sobel_mag = cv2.convertScaleAbs(sobel_mag)

                    # Threshold edges
                    _, thresh = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)

                    # Find contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Draw yellow outline over contours
                    cv2.drawContours(roi, contours, -1, (255, 200, 150), 2)  

                    # Put ROI back into frame
                    frame[y1:y2, x1:x2] = roi

        # ==== Save output ====
        if is_video:
            out.write(frame)
        else:
            cv2.imwrite(output_filepath, frame)
            break

    if is_video:
        cap.release()
        out.release()
    
    return output_filepath