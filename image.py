
import matplotlib.pyplot as plt # Import matplotlib for plotting images



# --- Helper function to display a single digit image ---
def display_digit(image_vector, actual_label, predicted_label=None):
    """
    Displays a single 784-element image vector as a 28x28 grayscale image.

    Args:
      image_vector (ndarray): A 1D numpy array of 784 pixel values.
      actual_label (int): The true label of the digit.
      predicted_label (int, optional): The predicted label of the digit by the model.
                                       If provided, it's included in the title.
    """
    # Reshape the 1D vector (784) back into a 2D image (28x28)
    image_2d = image_vector.reshape(28, 28)

    plt.figure(figsize=(2, 2)) # Small figure size for a single digit
    plt.imshow(image_2d, cmap='gray') # Display as grayscale image
    plt.axis('off') # Hide axes

    title = f"Actual: {actual_label}"
    if predicted_label is not None:
        title += f"\nPredicted: {predicted_label}"
    plt.title(title)
    plt.show() 



# Print shapes of the loaded and preprocessed data
print(f"Loaded and preprocessed X shape: {X.shape}")
print(f"Loaded and preprocessed y shape: {y.shape}")

# Get number of training examples and features after loading and preprocessing
m, n = X.shape # m = 60000, n = 784
num_classes = 10 # Digits 0-9

# Print first element of loaded X and y for demonstration
print(f"\nFirst element of loaded X (first 10 pixels, normalized): {X[0,:10]}")
print(f"First element of loaded y: {y[0,0]}")

# Select a single random image from the (preprocessed) training set for prediction
random_index = np.random.randint(m)
# The image vector is already flattened in X. We reshape it to (1, n) for model.predict().
image_for_prediction_input = X[random_index].reshape(1, n)
actual_label = y[random_index, 0]

# --- Display the actual image before prediction ---
print(f"Displaying the image that was randomly selected for prediction:")
display_digit(X[random_index], actual_label)


# Use model.predict() to get the raw linear outputs (logits)
prediction_logits = model.predict(image_for_prediction_input)

print(f"\nRaw prediction logits: \n{prediction_logits}")

# To get probabilities, apply the softmax function to the logits.
prediction_probabilities = tf.nn.softmax(prediction_logits).numpy()

print(f"\nPrediction probabilities (softmax output): \n{prediction_probabilities}")
print(f"Sum of prediction probabilities: {np.sum(prediction_probabilities):0.3f}")

# To determine the predicted digit, find the index of the highest probability.
predicted_digit = np.argmax(prediction_probabilities)

print(f"\nPredicted digit (index with highest probability): {predicted_digit}")

# --- Display the image again, now with the prediction ---
print(f"\nDisplaying the image with its actual and predicted label:")
display_digit(X[random_index], actual_label, predicted_digit)
