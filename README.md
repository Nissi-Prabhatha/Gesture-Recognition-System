
# 🤖 Real-Time Hand Gesture Recognition System

A real-time hand gesture recognition system using **Convolutional Neural Networks (CNN)** for intuitive and contactless human-computer interaction. The system processes webcam input to recognize hand gestures and perform corresponding control actions.

---

## 📌 Objective

To build a computer vision application that detects and classifies real-time hand gestures using webcam input, and uses those gestures to perform basic system control operations, enhancing accessibility and hygiene-based interaction.

---

## 🧠 Technologies Used

- **Python 3.8**
- **TensorFlow**
- **OpenCV**
- **Scikit-learn**
- **h5py**
- **Eel (for GUI integration)**

---

## 🖐️ Recognized Gestures

The system recognizes the following 9 gestures:
- One
- Two
- Three
- Four
- Five
- Palm
- Fist
- OK
- None

Each gesture is associated with a specific action or command within the application.

---

## 🧪 Dataset

- **Total Gestures**: 9 categories
- **Images per Category**: ~700
- **Total Images Used**: 773 for testing (randomly distributed)

---

## 🧮 Model Comparison

| Model                     | Recall (%) | F1 Score (%) | Precision (%) |
|--------------------------|------------|---------------|----------------|
| **CNN (Used)**           | 95.28      | 93.46         | 94.21          |
| Linear Regression (LR)   | 51.70      | 56.50         | 64.04          |
| LDA                      | 51.61      | 56.56         | 65.38          |
| Decision Tree Classifier | 53.68      | 59.44         | 67.47          |
| Random Forest            | 52.22      | 57.59         | 69.75          |
| Gaussian Naive Bayes     | 12.54      | 16.78         | 31.19          |

---

## 🖼️ Sample Outputs

The application displays:
- Raw webcam input
- Region of interest (ROI) with gesture prediction
- Binary threshold image
- Movement vector (dX, dY) with direction
- Predicted gesture with confidence level

---

## 🔧 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/Nissi-Prabhatha/Gesture-Recognition-System.git
   cd Gesture-Recognition-System
