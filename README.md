AI DRIVEN ROAD DAMAGE DETECTION

Road surface monitoring is a critical aspect of maintaining transportation safety and infrastructure quality. Traditional manual inspection methods are time-consuming, labor-intensive, and prone to human error. To address these challenges, this project proposes a real-time AI-based system for detecting and analyzing road anomalies such as potholes, cracks, manholes, and speed bumps using computer vision and deep learning techniques.
The proposed method integrates YOLOv8, a state-of-the-art object detection model, with MiDaS, a monocular depth estimation model, to identify and evaluate road defects from live video streams or image inputs. YOLOv8 accurately localizes anomalies, while MiDaS generates a relative depth map to estimate the distance and depth of each detected object. A severity score is then computed using a hybrid formula combining relative area and inverse normalized depth, classifying anomalies into Safe (Low), Warning (Medium), and Danger (High) categories.
The system provides real-time visualization with color-coded bounding boxes and distance measurements (in cm), enhancing interpretability for users and authorities. Experimental results demonstrate that the proposed approach effectively detects multiple road anomalies under varying lighting and surface conditions with minimal latency, contributing to safer and more efficient road maintenance.


Output
<img width="958" height="477" alt="image" src="https://github.com/user-attachments/assets/30a4cc7a-df6e-44b3-b041-e3fb6d60cfbe" />

<img width="960" height="472" alt="image" src="https://github.com/user-attachments/assets/9e63b5c8-9950-4d69-8129-662883bbd11b" />

MANHOLE

<img width="959" height="397" alt="image" src="https://github.com/user-attachments/assets/20e4cb37-7b53-4f59-82e1-30dc3bfb197e" />

SPEED BUMP

<img width="952" height="392" alt="image" src="https://github.com/user-attachments/assets/c6019a8e-4b75-4751-9553-16d3d0c44c28" />

ROAD CRACKS

<img width="945" height="399" alt="image" src="https://github.com/user-attachments/assets/7dcceee3-12f9-43ab-beda-5b7a8b5869eb" />

POTHOLE

<img width="958" height="438" alt="image" src="https://github.com/user-attachments/assets/497630e3-4db0-4ded-a7d0-63b5cf03e4e0" />





