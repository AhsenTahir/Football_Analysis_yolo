import ultralytics 
from ultralytics import YOLO

# Use the correct path to your video file in Google Drive
model = YOLO("models/best_github.pt")
results = model.predict("/videos/1.mp4", save=True)

print(results)

for box in results[0].boxes:
    print("***************************************************************")
    print(box)
