# Real-Time Hand Sign Digit Recognition (0–9)

A complete computer vision pipeline to collect, train, and predict hand-signed digits using OpenCV, TensorFlow, and MediaPipe.

This project lets you:
- Collect your own digit dataset using a webcam
- Train a CNN model to recognize digits (0–9) from grayscale hand images
- Predict digits in real time using hand detection and classification

---

## Features

- Webcam-based dataset collection  
- Convolutional Neural Network (CNN) training  
- MediaPipe hand detection  
- Real-time digit prediction with smoothing  
- Live confidence score overlay

---

## Dependencies

Install the required Python libraries:
```
pip install tensorflow opencv-python mediapipe scikit-learn matplotlib numpy
```
---

## Project Structure:

```
cv-project/
├── collect_dataset.py     # Script to collect hand sign images
├── train_model.py         # Script to train the CNN
├── camera_handdetect.py   # Real-time prediction using webcam
├── model.h5               # Trained TensorFlow model (generated)
├── Dataset/               # Folder with images per digit class
└── README.md              # This file
```

## Step-by-Step Usage

### 1 - Collect Your Dataset

Run:
```
python collect_dataset.py
```
- A webcam window will open.
- A green rectangle (ROI) shows the area captured.
- Press keys 0 to 9 to save images for that digit.
- Press 'q' to quit.
- Images will be saved in:
  ``` Dataset/0, Dataset/1, ..., Dataset/9 ```

### 2 - Train the CNN Model

After collecting enough images (at least 100 per class), run:
``` 
python train_model.py
```

- Loads and preprocesses all images.
- Trains a CNN model.
- Saves the model as:
  ``` model.h5 ```

### 3 - Real-Time Prediction

Run:
``` 
python camera_handdetect.py
```

- Starts the webcam and detects a hand using MediaPipe.
- Predicts the digit using the trained CNN.
- Displays the digit and confidence score on screen.
- Press 'q' to quit.

## CNN Model Architecture

- Input: 100x100 grayscale image  
- → Conv2D(32) + MaxPooling  
- → Conv2D(64) + MaxPooling  
- → Flatten  
- → Dense(512) + Dropout  
- → Dense(128) + Dropout  
- → Dense(10) with Softmax output

---

## Notes

- You can retrain the model any time after collecting more data.
- Try to collect samples with consistent lighting and angles.
- The prediction buffer reduces noise by smoothing results over recent frames.

---

## Authors
**Kartik Vanjani**,
**Yana Sivakova**, 
**Kirill Kheyfets**,
**Kristofer Deda**  
