 ReLU Activation ---
# The Rectified Linear Unit (ReLU) activation function is defined as a = max(0, z).
# It's commonly used in the hidden layers of neural networks to introduce non-linearity.
# This allows the network to learn more complex patterns than it could with only linear functions.

 - Softmax Function ---
# my_softmax: A NumPy implementation of the softmax function.
# Softmax converts a vector of arbitrary real values into a probability distribution.
# Each output value will be between 0 and 1, and the sum of all outputs will be 1.
# This is ideal for the output layer of a multiclass classification problem.

 Dataset (Loading MNIST):
#  will load the actual MNIST handwritten digit dataset.
# MNIST consists of 60,000 training images and 10,000 test images.
# Each image is 28x28 pixels in grayscale.
Reshape images: MNIST images are 28x28. The model expects a 1D vector.
#    We will flatten the 28x28 images to 784 features (28 * 28).
#    The input layer of the model will also be updated to (784,).
# The original images are (num_images, 28, 28). We want (num_images, 784).


  # Output Layer: A Dense layer with 10 units, one for each digit (0-9).
        # 'linear' activation is used here. This means the output values (logits)
        # are directly passed without transformation.
        # The softmax activation will be applied by the loss function during training
        # for numerical stability (SparseCategoricalCrossentropy with from_logits=True).

        # Output Shape: (None, X) where None indicates the batch size (can be any).
# Param #: Number of weights and biases for that layer.
# 4.5 Softmax Placement and Model Compilation
# During training, it's more numerically stable to apply softmax together with the loss function.
# This is indicated by `from_logits=True` in the loss definition.
# `SparseCategoricalCrossentropy` is suitable for integer labels (0, 1, 2... for classes).

   # when labels are integers (e.g., 0, 1, 2 for classes).
    # `from_logits=True` tells the loss function that the model's output layer
    # provides raw unscaled values (logits) and it should apply softmax internally.

     # Optimizer: Adam (Adaptive Moment Estimation) is a popular and effective optimizer.
    # learning_rate: Controls the step size during weight updates.

# Metrics to monitor during training (optional, but good for understanding performance)

# You can plot the training loss and accuracy over epochs using `history.history`
# For example:
# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()
#
# plt.plot(history.history['accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()

