# Dataset Card - Joan Hand Gesture Dataset - Miniproject for MLME

The full training dataset is not included in the project submission due to size constraints.
To access the data, visit [the public link to my Edge Impulse project](https://studio.edgeimpulse.com/studio/828181), where all samples can be viewed and downloaded directly in their original format.

## 1. Dataset Name
- **Title:** Joan Hand Gesture Dataset  
- **Version:** 1.0  
- **Author:** Joan Montés Mora  
- **Year:** 2025 

## 2. General Description
This dataset contains RGB images of four static hand gestures (`fist`, `five`, `okay`, `two`) captured with a laptop webcam.  
Its purpose is to train an object detection model (YOLO-Pro) enabling real-time gesture-based media control.

**Data type:** RGB images  
**Intended task:** Real-time gesture detection  
**File format:** PNG

## 3. Motivation
The dataset was created to develop a gesture-based user interface for controlling a media player without physical interaction.  

## 4. Dataset Composition
- **Total samples:** 403 images  
- **Samples per class:** aprox 100 per gesture  
- **Classes:**  
  - `fist` → next song  
  - `five` → play/pause  
  - `okay` → volume up  
  - `two` → volume down  
- **Attributes:**  
  - RGB image  
  - Bounding box  
  - Class prediction value    
- **Image format:** 320×320 RGB

## 5. Source
- **Origin:** Fully self-collected 
- **Device:** Laptop integrated webcam  
- **Environment:** Indoor, artificial lighting   

## 6. Collection & Processing Methodology
- **Collection method:** 
  - Variations in background, distance, orientation, and left/right hand  
- **Preprocessing:**  
  - Resize to 320×320  
  - RGB conversion  
  - Normalization to [0,1]  
- **Cleaning:**  
  - Removal of blurry frames  
  - Manual verification of labels  

## 7. Ethical Considerations & Risks
- **Biases:**  
  - Single-user dataset (limited demographic diversity)  
  - Indoor-only lighting conditions  
  - Visual similarity between `okay` and `two` cause ambiguity  
- **Privacy:**  
  - No faces or identifiable personal features included  

## 8. License & Permitted Use
- **License:** Academic and experimental use only  
- **Restrictions:**  
  - Not intended for commercial use  
  - Not intended for redistribution  

## 9. Dataset Splits
- **Train / Validation / Test:**  
- Train: ~80%  
- Validation/Test: ~20% (automatic split in Edge Impulse)  

## 10. Results
- **Task:** Object detection
- **Reference model:** YOLO-Pro (pico, 682K parameters)  
- **Performance metrics:**  
- Test accuracy: 94.67%  
- mAP@50: 1.00  
- mAP@75: 0.92  
- Recall: 0.77  

