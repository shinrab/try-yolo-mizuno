from ultralytics import YOLO

# Load a pretrained YOLO model.
model = YOLO("yolo26n.pt")

# Run inference once on a local image and save outputs under runs/detect/predict*
results = model.predict(source="bus.jpg", save=True, conf=0.25)

detected = len(results[0].boxes)
print(f"inference completed: {detected} objects detected")