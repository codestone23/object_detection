from keras.models import load_model
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from keras.preprocessing import image

np.set_printoptions(suppress=True)
# Tải mô hình được đào tạo trước từ tệp
model = load_model('keras_model.h5', compile=False)
# đọc file text chứa những tên những đôi tượng đã huấn luyện
class_labels = open("labels.txt", "r", encoding='UTF8').readlines()

# Sử dụng camera laptop
cap = cv2.VideoCapture(0)

# kết nối camera điện thoại bằng phần mềm DroidCamApp laptop
# cap = cv2.VideoCapture(1)

# kết nối đến camera điện thoại bằng url
# ip_address = "10.21.32.91"
# url = f"http://{ip_address}:4747/video"
# cap = cv2.VideoCapture(url)

while True:
    # Chụp khung hình từ webcam
    ret, frame = cap.read()
    # Xử lý trước khung để phân loại
    img = cv2.resize(frame, (224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    # Sử dụng mô hình để phân loại khung
    prediction = model.predict(img_tensor)
    # hiện chữ lên màn hình
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
    # Thoát vòng lặp nếu người dùng nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release webcam và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()