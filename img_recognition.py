from keras.models import load_model
from PIL import ImageFont,Image, ImageOps
import numpy as np
import cv2  # Import OpenCV

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r", encoding='UTF8').readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Initialize OpenCV window
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# Replace this with the path to your image
image_path = "laptop.jpg"

while True:
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    # Display the image with the class name and confidence score
    image_with_text = cv2.putText(image_array, f"Class: {class_name[2:]}, Confidence Score: {confidence_score:.4f}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", image_with_text)

    # Check for 'Q' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break

# Release OpenCV window and close it
cv2.destroyAllWindows()
