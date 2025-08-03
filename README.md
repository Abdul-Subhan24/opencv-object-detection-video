# 🧠 Object Detection with OpenCV and SSD MobileNet

This project demonstrates how to perform **object detection on a video** using **OpenCV's DNN module** with a pre-trained **SSD MobileNet v3 model**. It detects common everyday objects (like people, cars, bottles, etc.) in each frame of a video and highlights them with **bounding boxes and labels**.

---

## 📌 Project Features

- Uses a pre-trained model trained on the **COCO dataset** (80 object classes)
- Detects objects in a **video file** frame-by-frame
- Draws **bounding boxes** with confidence percentages
- Displays the first 10 frames with object annotations
- Ideal for beginners to understand how object detection works using OpenCV

---

## 📂 Files Used in This Project

| File Name                      | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `frozen_inference_graph.pb`   | Pre-trained model file (weights) used for detecting objects                 |
| `Object Detection.pbtxt.txt`  | Configuration file that tells OpenCV how to interpret the model             |
| `Object Detection Labels.txt` | Text file containing 80 COCO class names, one per line                      |
| `Object Detection test.mp4`   | Sample video file used as input for detection                               |
| `object_detection.py`         | The main Python file that loads the model, processes the video, and shows results |

---

## ▶️ How It Works (Step-by-Step)

1. **Load the pre-trained model** using `cv2.dnn_DetectionModel`  
2. **Read the class labels** (like 'person', 'car', etc.) from a text file  
3. **Open the input video**  
4. **For the first 10 frames only**:
   - Pass the frame to the model
   - Get detected objects, bounding boxes, and confidence scores
   - Draw rectangles and label text on the frame
   - Show the frame using `matplotlib`
5. Release the video file when done

---

## 🧠 What Is SSD MobileNet?

- **SSD (Single Shot Detector)**: A fast and efficient object detection model
- **MobileNet v3**: A lightweight CNN optimized for mobile and real-time applications
- Trained on the **COCO dataset**, which contains 80 common objects (person, dog, bottle, etc.)

---

## 📋 Requirements

```bash
pip install opencv-python matplotlib
```bash
pip install opencv-python matplotlib
