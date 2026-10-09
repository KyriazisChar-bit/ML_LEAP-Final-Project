# ML_LEAP-Final-Project
This code implements and evaluates Convolutional Neural Networks (CNNs) for handwritten digit recognition using the MNIST dataset. It investigates how altering model depth—specifically by adding an extra convolutional layer—impacts the network's ability to extract spatial hierarchies and classify complex handwritten patterns.
# Handwritten Digit Recognition with CNNs (MNIST)

A Deep Learning project built with **Python**, **TensorFlow/Keras**, and **scikit-learn** to classify handwritten digits using Convolutional Neural Networks (CNNs) on the classic **MNIST dataset**.

## 🚀 Project Overview
This project explores the design, training, and evaluation of image classification models for handwritten digits. It follows an iterative approach—starting with a baseline Convolutional Neural Network and experimenting with network depth (adding convolutional layers) to analyze performance improvements, loss reductions, and error patterns.

## 🛠️ Tech Stack & Libraries
* **Python** 
* **TensorFlow & Keras** (Deep Learning framework)
* **scikit-learn** (Dataset loading, data splitting, evaluation metrics)
* **Matplotlib** (Data visualization, loss/accuracy curves, confusion matrices)

## 📊 Key Steps & Methodology
1. **Data Preparation:** Fetched the MNIST dataset (`mnist_784`), normalized pixel values to `[0, 1]`, reshaped images into `(28, 28, 1)`, and performed a stratified train/test split.
2. **Model 1 (Baseline CNN):** Built a sequential architecture featuring a `Conv2D` layer, `MaxPooling2D`, `Dropout (0.3)` for regularization, a `Flatten` layer, and Dense layers.
3. **Training & Optimization:** Trained using the Adam optimizer (`sparse_categorical_crossentropy` loss) combined with an **Early Stopping** callback (`val_loss` monitoring) to prevent overfitting.
4. **Evaluation & Visualization:** Assessed performance using Test Accuracy, Loss curves, and detailed **Confusion Matrices**.
5. **Model Iteration (Deeper CNN):** Implemented a second model (Option A) by adding an extra convolutional block to study how increased depth impacts feature extraction and generalization.

## 📈 Results & Findings
* **Baseline Model (Model 1):** Achieved a test accuracy of **98.7%** with a test loss of **0.046**.
* **Doped/Deeper Model (Model 2):** Improving architecture depth increased test accuracy to **99.0%** and reduced test loss to **0.035**.
* **Key Takeaway:** Adding depth allowed the network to learn more complex spatial hierarchies, reducing overall error rates by ~23% and increasing prediction confidence without causing overfitting.
