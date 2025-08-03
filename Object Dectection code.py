import cv2
import matplotlib.pyplot as plt

# File paths
frozen_model = 'frozen_inference_graph.pb'
pbtxt_file = 'Object Detection.pbtxt.txt'
file_name = 'Object Detection Labels.txt'
video_file = 'Object Detection test.mp4'  # ✅ Add this line

# Load model
model = cv2.dnn_DetectionModel(frozen_model, pbtxt_file)

# Load class labels
with open(file_name, 'r') as f:
    class_labels = f.read().rstrip('\n').split('\n')

# Configure model
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Load video
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    raise IOError("❌ Cannot open video file")

# Show only first 10 frames for demo
frame_count = 1
while frame_count <= 10:
    ret, frame = cap.read()
    if not ret:
        break

    class_index, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(class_index) != 0:
        for class_idx, conf, box in zip(class_index.flatten(), confidence.flatten(), bbox):
            label = f'{class_labels[class_idx - 1]} : {round(conf * 100, 2)}%'
            cv2.rectangle(frame, box, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, label, (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Convert BGR to RGB for display in Colab
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(f'Frame {frame_count}')
    plt.show()

    frame_count += 1

cap.release()
print("✅ Displayed 10 frames with object detection.")