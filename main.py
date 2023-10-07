from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from keras.preprocessing import image

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r", encoding='UTF8').readlines()
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    # cv2.imshow("Webcam Image", img)
    # image = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    # image = (image / 127.5) - 1
    image = np.asarray(img, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    prediction = model.predict(image)
    print(list(prediction[0]))
    for predictions in (list(prediction[0])):
        if predictions >= 0.8:
            name = class_names[np.argmax(prediction)]
            fontpath = "./Roboto-Bold.ttf"
            font = ImageFont.truetype(fontpath, 32)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((20, 20), name[2:], font=font, fill=(0, 255, 0, 1))
            img = np.array(img_pil)
            # cv2.putText(img,name[2:],(5,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,250,0),1)
            print(class_names[np.argmax(prediction)])
    cv2.imshow("Webcam Image", frame)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break
camera.release()
cv2.destroyAllWindows()
