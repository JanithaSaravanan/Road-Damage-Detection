from ultralytics import YOLO
import cv2
import time
import geocoder
import os

# ==== 1. Load YOLOv8 model ====
model = YOLO("project_files/best.pt")   # <-- your .pt file

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

    results = model(frame, conf=0.5)  # run YOLOv8 detection

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bbox
            conf = float(box.conf[0])
            label = "pothole"

            # Draw detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f} {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save coordinates + image every few seconds
            if i == 0 or (time.time() - b) >= 2:
                cv2.imwrite(os.path.join(result_path, f'pothole{i}.jpg'), frame)
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
