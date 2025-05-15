# 😷 Face Mask Detection with YOLOv7

This project focuses on fine-tuning the **YOLOv7 object detection framework** to detect **face mask compliance** in real-world environments. The model identifies three distinct classes:

- ✅ Correctly Worn Mask  
- ❌ Incorrectly Worn Mask  
- 🚫 No Mask  

Developed in collaboration with **Yashwanth**, this project aims to build a reliable, real-time solution for monitoring face mask usage using state-of-the-art deep learning techniques.

---

## 📌 Project Highlights

- 🧠 Fine-tuned YOLOv7 for a **3-class detection task**
- 🖼️ Custom dataset sourced from Kaggle and **manually annotated using CVAT**
- 📊 Achieved **mAP@0.5 of 74%** and **Precision > 85%**
- 📈 Visualized model performance using TensorBoard
- 🔁 Easily extendable and reproducible with provided training/inference scripts

---

## 🔧 Project Workflow

### 1️⃣ Base Model Pretraining
- Trained YOLOv7 on the **COCO dataset** for general object detection
- Established a strong base model to improve downstream task performance

### 2️⃣ Custom Data Collection & Annotation
- Used a **Kaggle face mask dataset**
- Manually annotated images using **CVAT** (Computer Vision Annotation Tool) to create high-quality labels for:
  - `correct_mask`
  - `incorrect_mask`
  - `no_mask`

### 3️⃣ YOLOv7 Fine-Tuning
- **Modified architecture** for 3-class setup
- Updated training configs:
  - **Image size:** 640x640  
  - **Batch size:** 8  
  - **Optimizer:** SGD with momentum  
  - **Augmentations:** Mosaic, random horizontal flips, and scale jittering
- Monitored training using **TensorBoard**

---

## 📈 Model Performance

| Metric              | Score              |
|---------------------|--------------------|
| **mAP@0.5**         | ~74%               |
| **mAP@0.5:0.95**    | ~48%               |
| **Precision**       | > 85%              |
| **Recall**          | > 70%              |

The results show strong and consistent performance, with accurate detections across a range of IoU thresholds.

---

## 🖼️ Sample Output

### 📸 Detection Image

![Detection Output](img2.png)

### 🎬 Detection Video

▶️ [Watch Detection Demo](road%20(2).mp4)

> Tip: If the video doesn't autoplay in GitHub, download or open it in a new tab.

---

## 🚀 Getting Started

### 📥 Clone the Repository
```bash
git clone https://github.com/snuka75/Yolov7.git
cd Yolov7
```

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🏋️ Train the Model
```bash
python train.py --img 640 --batch 8 --epochs 100 --data face-mask.yaml --cfg yolov7-face_mask.yaml --weights yolov7.pt
```

### 🎯 Run Inference
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/image_or_video
```

---

## 🛠️ Tech Stack

| Component     | Purpose                           |
|---------------|------------------------------------|
| **YOLOv7**     | Object detection architecture     |
| **CVAT**       | Custom image annotation tool      |
| **PyTorch**    | Model training and evaluation     |
| **TensorBoard**| Visual performance monitoring     |
| **OpenCV**     | Image processing during inference |

---

## 🗃️ Dataset

- 📦 Source: [Kaggle Face Mask Dataset](https://www.kaggle.com/)
- 🛠️ Labeling: Manual annotation via **CVAT**
- 📁 Format: YOLOv7-compatible with `.txt` label files for each image

---

## 🙌 Collaborators

- **Samhitha Nuka**  
- **Yashwanth Vijayaragavan**

---

## 📄 License

This project is licensed under the MIT License.

---

## 💡 Future Improvements

- 🔁 Integrate real-time webcam inference
- 📦 Convert to ONNX or TensorRT for edge deployment
- 📊 Add a dashboard for live analytics and alerting
