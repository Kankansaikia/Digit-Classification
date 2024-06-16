# Digit Recognition with Neural Networks

This project demonstrates how to build a simple neural network using TensorFlow and Keras to recognize hand-written digits from the MNIST dataset.

## Overview

The project covers the following key aspects:

- **Loading and Exploring the Dataset**: Loading the MNIST dataset using Keras and exploring its structure.
- **Data Preprocessing**: Scaling the pixel values of images to improve model training efficiency.
- **Building the Neural Network**: Designing a neural network with dense layers for digit classification.
- **Model Compilation and Training**: Compiling the model with appropriate loss and metrics, then training it on the training set.
- **Model Evaluation**: Evaluating the trained model on the test set to measure its accuracy.
- **Prediction**: Using the trained model to predict digits on new unseen images.

## Prerequisites

To run this project, you need the following Python libraries:

- numpy
- PIL (Pillow)
- matplotlib
- seaborn
- OpenCV (cv2)
- TensorFlow
- Keras

You can install these libraries using pip:

```bash
pip install numpy pillow matplotlib seaborn opencv-python tensorflow keras
```


## Files Included

- **digit_recognition.py**: This Python script contains the entire pipeline for loading data, preprocessing, building, training, and making predictions with the neural network.
  
- **Mnist.png**: This sample image file is used to demonstrate the predictive system. You can use this image or replace it with your own to test the model's prediction capabilities.
  
- **README.md**: This file provides an overview of the project, installation instructions, and usage guidelines.



## Additional Notes

- **Model Architecture**: The neural network architecture comprises three layers:
  - Two hidden layers with ReLU activation.
  - One output layer with softmax activation for digit classification.
  
- **Data Scaling**: Pixel values of images are scaled to a range of 0 to 1. This preprocessing step enhances model convergence during training.
  
- **Prediction System**: The script includes a predictive system that enables input of a new image (`Mnist.png` in this case) to obtain a prediction of the digit it represents.





