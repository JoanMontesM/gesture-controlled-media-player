# Gesture-Controlled Media Player
**Using Edge Impulse Hand Gesture Recognition**

This repository contains the implementation of a **gesture-controlled media player** that uses a machine learning model to recognize hand gestures in real time and map them to media playback controls.

The project was developed as part of the *Machine Learning for Media Experiences* miniproject at **Aalborg University**.

---

## Project Overview

The goal of this project is to design and implement a complete interactive ML system capable of:

- Detecting **four hand gestures** in real time  
- Mapping each gesture to a media player action  
- Running efficiently on consumer hardware using edge-optimized models  

The system uses:
- A **self-collected image dataset**
- A **YOLO-Pro object detection model** trained with **Edge Impulse**
- A **Python real-time application** using TensorFlow Lite, OpenCV, and Pygame

---

## Supported Gestures & Actions

| Gesture | Media Action |
|------|-------------|
| Fist | Next track |
| Five (open hand) | Play / Pause |
| Okay | Volume up |
| Two (V sign) | Volume down |

To avoid accidental triggers, an action is executed only when the same gesture is detected consistently over **10 consecutive frames**.

---

## Dataset

- Fully **self-collected dataset**
- ~100 images per gesture
- Captured using a laptop webcam
- Indoor environment with artificial lighting
- Variations in:
  - Backgrounds
  - Camera distance
  - Hand orientation (left & right)

All images were manually cleaned, selected, and labeled into four gesture classes.

---

## Model Design & Training

- **Platform:** Edge Impulse  
- **Model:** YOLO-Pro (object detection)
- **Input size:** 320 × 320 RGB images  
- **Model size:** Pico (~682k parameters)  
- **Training cycles:** 100  
- **Learning rate:** 0.01  
- **Pretrained weights:** Enabled  

### Performance (Test Set)

- Accuracy: **94.67%**
- mAP@50: **1.00**
- mAP@75: **0.93**
- Recall: **~0.76**

The training and validation loss curves indicate good generalization with no clear signs of overfitting.

---

## System Architecture

1. Webcam captures frames in real time  
2. Frames are preprocessed:
   - Resized to 320×320
   - Converted to RGB
   - Normalized
3. TensorFlow Lite interpreter performs inference
4. Valid detections above a confidence threshold are processed
5. Gesture actions are triggered after temporal consistency
6. OpenCV displays:
   - Bounding boxes
   - Detected gesture
   - Current playback state
7. Pygame controls the media player logic

---

## How to Run

### Requirements

- Python 3.x  
- Webcam  
- Required libraries:
  - `tensorflow` or `tflite-runtime`
  - `opencv-python`
  - `pygame`
  - `numpy`

### Basic Execution

```bash
python run_model.py
```
Make sure the exported TensorFlow Lite model from Edge Impulse is correctly placed in the project directory.

---

## Author
**Joan Montés Mora**
Aalborg University
Machine Learning for Media Experiences









## How to RunJoan Montés Mora
Aalborg University
Machine Learning for Media Experiences
