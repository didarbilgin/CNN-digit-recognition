# Handwritten Digit Recognition using CNN

This project demonstrates a Convolutional Neural Network (CNN) built with TensorFlow/Keras to recognize handwritten digits from the MNIST dataset. Additionally, it allows users to upload and classify their own digit images using the trained model.

## Overview

- Dataset: MNIST Handwritten Digits
- Model: Convolutional Neural Network (Conv2D, MaxPooling2D, Flatten, Dense)
- Evaluation: Accuracy & loss visualization on test data
- Prediction: Custom digit image input via OpenCV and visualization via Matplotlib

## Tech Stack

- Python 3.9
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib

## Model Training & Accuracy

The model was trained for 5 epochs with validation, achieving approximately 98% accuracy on test data:

```
Epoch 5/5
accuracy: 0.9812 - loss: 0.1353 - val_accuracy: 0.9728 - val_loss: 0.3865
Test Accuracy: 0.9751
Test Loss: 0.3441509008407593
```

## Prediction on Custom Input

You can upload your own handwritten digit image (e.g., a PNG image), and the model will:

- Resize it to 28x28 grayscale
- Normalize and reshape it for input
- Predict the digit class (0-9)
- Output confidence scores for each class

### Visualization Example

Below is an example showing:
1. Original image
2. Resized 28x28 image with prediction
3. Confidence scores per digit class

![Prediction Visualization](https://github.com/user-attachments/assets/c9941598-f9a0-46d7-839b-ba5b41e02f2a)

### Input Digit Example
![Input digit](https://github.com/user-attachments/assets/54174b64-1ea0-4da1-958a-f32eb2e014b6)


## How to Run

1. Clone the repository
2. Train the model in main.py (if not using the provided `model.h5`)
3. Run the prediction script:

```
python model_pred.py
```

Ensure your custom digit image is named `digit.png` and is in the same directory.

## Files

- `main.py`: Model training script
- `model.h5`: Trained model
- `model_pred.py`: Prediction on custom input
- `digit.png`: Sample input image

## License

This project is licensed under the MIT License.
