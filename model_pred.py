import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Load the model
model = tf.keras.models.load_model("model.h5")

# Load the image
original_img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)

# Resize to 28x28
img = cv2.resize(original_img, (28, 28))
img = img / 255.0
img_input = img.reshape((1, 28, 28, 1))  # For model input

# Make prediction
predictions = model.predict(img_input)
predicted_class = np.argmax(predictions)

# Print results to console
print(f"Predicted Digit: {predicted_class}")
print("Confidence scores for each class:")
for i, score in enumerate(predictions[0]):
    print(f"{i}: {score:.4f}")

# Visualize
plt.figure(figsize=(12, 4))

# 1. Original image
plt.subplot(1, 3, 1)
plt.imshow(original_img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# 2. Resized image (28x28)
plt.subplot(1, 3, 2)
plt.imshow(img, cmap="gray")
plt.title(f"Resized 28x28\nPrediction: {predicted_class}")
plt.axis("off")

# 3. Bar chart of confidence scores
plt.subplot(1, 3, 3)
plt.bar(range(10), predictions[0])
plt.title("Confidence Scores")
plt.xlabel("Digit Class")
plt.ylabel("Probability")
plt.xticks(range(10))

plt.tight_layout()
plt.show()