from ultralytics import YOLO
import cv2
import time
import geocoder
import os
import torch
import torchvision.transforms as T

# ==== 1. Load YOLOv8 and MiDaS models ====
model = YOLO("project_files/best.pt")  # YOLOv8 for object detection

# Load the MiDaS monocular depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# MiDaS image transformations
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# ==== 2. Open video file ====
cap = cv2.VideoCapture("test4.mp4")
ret, frame = cap.read()
if not ret:
    print("❌ Could not read video. Please check the file path.")
    exit()

height, width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
result = cv2.VideoWriter('result.mp4', fourcc, 10, (width, height))

# ==== 3. Output folder for potholes ====
result_path = "pothole_coordinates"
os.makedirs(result_path, exist_ok=True)

g = geocoder.ip('me')
starting_time = time.time()
frame_counter = 0
i = 0
b = 0

# ==== 4. Detection loop ====
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1

    # Get depth map from MiDaS model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    
    # Run YOLOv8 detection
    results = model(frame, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = "pothole"

            # Estimate distance from depth map
            pothole_region = depth_map[y1:y2, x1:x2]
            if pothole_region.size > 0:
                average_depth = pothole_region.mean()
                
                # Simple scaling for demonstration to convert relative depth to cm
                k = 5000  # Adjust this value for your specific scene
                distance_in_cm = (k / average_depth) * 100
                
                # Prepare text for display
                text = f"Pothole: {conf:.2f} | Dist: {distance_in_cm:.0f}cm"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw text with a background for better readability
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_color = (255, 255, 255) # White
                text_background_color = (0, 0, 0) # Black
                
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.rectangle(frame, (x1, y1 - text_height - baseline - 10), 
                              (x1 + text_width, y1 - 10), text_background_color, -1)
                
                cv2.putText(frame, text, (x1, y1 - baseline - 5),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Save coordinates + image every few seconds
            if i == 0 or (time.time() - b) >= 2:
                # Use a lower quality for smaller file size
                jpeg_quality = 75
                cv2.imwrite(os.path.join(result_path, f'pothole{i}.jpg'), frame, 
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                with open(os.path.join(result_path, f'pothole{i}.txt'), 'w') as f:
                    f.write(str(g.latlng))
                b = time.time()
                i += 1

    ending_time = time.time() - starting_time
    fps = frame_counter / ending_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    result.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()
print("✅ Video saved as result.mp4")