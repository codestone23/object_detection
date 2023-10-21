from keras.models import load_model
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from keras.preprocessing import image

np.set_printoptions(suppress=True)

# Tải mô hình được đào tạo trước từ tệp
model = load_model('keras_model.h5', compile=False)

# Camera phone
# Địa chỉ IP của camera IP Webcam trên điện thoại
ip_address = "10.21.32.91"

# Tạo URL kết nối đến camera
url = f"http://{ip_address}:4747/video"

# Initialize the webcam
class_labels = open("labels.txt", "r", encoding='UTF8').readlines()

# Sử dụng camera laptop
cap = cv2.VideoCapture(0)

# Sử dụng camera điện thoại thông qua phần mềm droidcamapp trên laptop
# cap = cv2.VideoCapture(1)

# Sử dụng trực tiếp camera điện thoại bằng url
# cap = cv2.VideoCapture(url)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    # Pre-process the frame for classification
    img = cv2.resize(frame, (224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    # Use the model to classify the frame
    prediction = model.predict(img_tensor)
    # show text
    for predictions in (list(prediction[0])):
        if predictions >= 0.7:
            name = class_labels[np.argmax(prediction)]
            fontpath = "./Roboto-Bold.ttf"
            font = ImageFont.truetype(fontpath, 32)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 30), name[2:], font=font, fill=(0, 255, 0, 1))
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            print(class_labels[np.argmax(prediction)])
    # Get the class label for the frame
    class_index = np.argmax(prediction[0])

    # Get the class label for the frame
    class_label = class_labels[class_index]
    confidence_score = prediction[0][class_index]
    print("Class:", class_label[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Display the frame with the class label
    cv2.imshow('Webcam', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()