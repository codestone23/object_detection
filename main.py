import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageOps
from keras.models import load_model
from keras.preprocessing import image
import os

# Nạp mô hình
model = load_model('keras_Model.h5', compile=False)

# Nạp nhãn của các lớp
class_names = open("labels.txt", "r", encoding='UTF8').readlines()

# Tạo mảng với đúng kích thước để đưa vào mô hình Keras
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Khởi tạo cửa sổ OpenCV
# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)


def classify_image(image_path, output_folder):
    # Nạp hình ảnh
    image = Image.open(image_path).convert("RGB")

    # Thay đổi kích thước hình ảnh ít nhất là 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Chuyển hình ảnh thành mảng numpy
    image_array = np.asarray(image)

    # Chuẩn hóa hình ảnh
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Nạp hình ảnh vào mảng
    data[0] = normalized_image_array

    # Dự đoán bằng mô hình
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Hiển thị hình ảnh với tên lớp và phần trăm tin cậy
    class_index = np.argmax(prediction[0])
    class_label = class_names[class_index]
    confidence_score = prediction[0][class_index]
    print("Class:", class_label[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Cài đặt font
    fontpath = "./Roboto-Bold.ttf"
    draw = ImageDraw.Draw(image)

    # Vị trí và màu văn bản
    font_size = 24
    font = ImageFont.truetype(fontpath, font_size)

    # Vẽ văn bản lên hình ảnh
    x, y = 60, 10
    text = "{} ".format(class_label[2:])
    text_color = (0, 0, 0)

    # Đường dẫn để lưu hình ảnh mới
    draw.text((x, y), text, fill=text_color, font=font)

    # Đường dẫn để lưu hình ảnh mới
    output_path = os.path.join(output_folder, filename)

    # Lưu hình ảnh
    image.save(output_path)

def classify_webcam(url):
    # Khởi tạo webcam
    cap = cv2.VideoCapture(url)

    while True:
        # Chụp một khung hình từ webcam
        ret, frame = cap.read()

        # Tiền xử lý khung hình để phân loại
        img = cv2.resize(frame, (224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        # Sử dụng mô hình để phân loại khung hình
        prediction = model.predict(img_tensor)

        for predictions in (list(prediction[0])):
            if predictions >= 0.7:
                name = class_names[np.argmax(prediction)]
                fontpath = "./Roboto-Bold.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 30), name[2:], font=font, fill=(0, 255, 0, 1))
                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                print(class_names[np.argmax(prediction)])
        # Lấy chỉ số nhãn lớp cho khung hình
        class_index = np.argmax(prediction[0])
        # Lấy nhãn lớp cho khung hình
        class_label = class_names[class_index]
        confidence_score = prediction[0][class_index]
        print("Class:", class_label[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Hiển thị khung hình với nhãn lớp
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Thoát vòng lặp nếu người dùng nhấn 'q'
    cap.release()
    cv2.destroyAllWindows()


# Yêu cầu người dùng chọn giữa phân loại hình ảnh và phân loại webcam
while True:
    choice = input("Choose an option (1 for image, 2 for webcam, q to quit): ")

    if choice == '1':
        image_folder = "img"
        output_folder = 'data/input'
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            classify_image(image_path, output_folder)
        cv2.destroyAllWindows()

    elif choice == '2':
        url = 0
        classify_webcam(url)

    elif choice == 'q':
        break

    else:
        print("Invalid option. Please choose 1 for image, 2 for webcam, or q to quit.")
