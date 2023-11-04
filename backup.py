import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageOps
from keras.models import load_model
from keras.preprocessing import image
import os
import tkinter as tk
from tkinter import filedialog

# Nạp mô hình
model = load_model('keras_Model.h5', compile=False)

# Nạp nhãn của các lớp
class_names = open("labels.txt", "r", encoding='UTF8').readlines()

# Tạo mảng với đúng kích thước để đưa vào mô hình Keras
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Khởi tạo cửa sổ OpenCV
# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

def create_button(parent, text, command):
    button = tk.Button(parent, text=text, command=command, padx=10, pady=10)
    button.configure(
        background='white',
        foreground='black',
        font=('Arial', 14),
        relief=tk.RAISED,
        borderwidth=4,
        highlightthickness=4,
    )
    button.pack(pady=(20, 0))
    return button
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
    text_color = (0, 255, 0)

    # Đường dẫn để lưu hình ảnh mới
    draw.text((x, y), text, fill=text_color, font=font)

    # Đường dẫn để lưu hình ảnh mới
    output_path = os.path.join(output_folder, os.path.basename(image_path))

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
root = tk.Tk()
root.title("Image and Video Classifier")

def on_image_classification():
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

    def is_image_file(filename):
        _, file_extension = os.path.splitext(filename)
        return file_extension.lower() in IMAGE_EXTENSIONS
    def choose_images():
        image_paths = filedialog.askopenfilenames(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")])
        if image_paths:
            for image_path in image_paths:
                classify_image(image_path, 'data/input')
    def choose_folder():
        image_folder = filedialog.askdirectory(title="Select a Folder")
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            if is_image_file(filename):
                classify_image(image_path, 'data/input')
    def on_return():
        main_menu_frame.pack()
        image_frame.pack_forget()

    main_menu_frame.pack_forget()
    image_frame = tk.Frame(root)
    image_frame.pack()

    images_button = create_button(image_frame, "Choose images", choose_images)
    folder_button = create_button(image_frame, "Choose folder", choose_folder)
    image_return_button = create_button(image_frame, "Back", on_return)

    images_button.pack()
    folder_button.pack()
    image_return_button.pack()
def on_webcam_classification():
    def on_camera_laptop():
        url = 0
        classify_webcam(url)
    def on_camera_droidcam():
        url = 1  # Choose phone camera (or change it to your configuration)
        classify_webcam(url)
    def on_camera_phone():
        def start_camera():
            ip = ip_entry.get()
            port = port_entry.get()
            url = f"http://{ip}:{port}/video"
            classify_webcam(url)
        def on_return():
            webcam_frame.pack()
            webcam_frame_new.pack_forget()

        webcam_frame.pack_forget()
        webcam_frame_new = tk.Frame(root)
        webcam_frame_new.pack()

        ip_label = tk.Label(webcam_frame_new, text="Enter IP Address:")
        ip_label.pack()
        ip_entry = tk.Entry(webcam_frame_new)
        ip_entry.pack()

        port_label = tk.Label(webcam_frame_new, text="Enter Port:")
        port_label.pack()
        port_entry = tk.Entry(webcam_frame_new)
        port_entry.pack()

        start_button = create_button(webcam_frame_new, "Start Camera", start_camera)
        return_button = create_button(webcam_frame_new, "Back", on_return)

        start_button.pack()
        return_button.pack()
    def on_return():
        main_menu_frame.pack()
        webcam_frame.pack_forget()

    main_menu_frame.pack_forget()
    webcam_frame = tk.Frame(root)
    webcam_frame.pack()

    camera_laptop_button = create_button(webcam_frame, "Camera Laptop", on_camera_laptop)
    camera_phone_button = create_button(webcam_frame, "Camera DroidCam", on_camera_droidcam)
    camera_phone_button = create_button(webcam_frame, "Camera Phone", on_camera_phone)
    return_button = create_button(webcam_frame, "Back", on_return)

    camera_laptop_button.pack()
    camera_phone_button.pack()
    return_button.pack()

main_menu_frame = tk.Frame(root)
main_menu_frame.pack()

image_classification_button = create_button(main_menu_frame, "Image Classification", on_image_classification)
webcam_classification_button = create_button(main_menu_frame, "Webcam Classification", on_webcam_classification)
quit_button = create_button(main_menu_frame, "Quit", root.destroy)

image_classification_button.pack()
webcam_classification_button.pack()
quit_button.pack()

root.geometry('400x400')  # Set the initial window size
root.mainloop()