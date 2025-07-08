import tensorflow as tf
import matplotlib.pyplot as plt

# load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# reshape for CNN
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

# sequential model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (28,28,1)),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        tf.keras.layers.Conv2D(64, kernel_size = (2,2), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ]
)

model.summary()
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# fit the model
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
model.save("model.h5")
