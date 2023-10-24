from keras.models import load_model
from PIL import ImageFont, Image, ImageOps, ImageDraw
import numpy as np
import cv2  # Import OpenCV
import os

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

# Đường dẫn đến thư mục lưu ảnh cần nhận diện
image_folder = "img"

# Đường dẫn đến thư mục để lưu các ảnh đã nhận diện
output_folder = 'data/input'

# Tạo thư mục đầu ra nếu nó không tồn tại
os.makedirs(output_folder, exist_ok=True)

def Xuly(image_path,output_folder):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image1 = cv2.imread(image_path)
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

    class_index = np.argmax(prediction[0])
    # Get the class label for the frame
    class_label = class_names[class_index]
    confidence_score = prediction[0][class_index]
    print("Class:", class_label[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Đường dẫn đến tệp font TTF
    fontpath = "./Roboto-Bold.ttf"  # Thay đổi đường dẫn đến font TTF của bạn
    draw = ImageDraw.Draw(image)
    # Đối tượng font từ font TTF
    font_size = 24
    font = ImageFont.truetype(fontpath, font_size)

    # Vị trí bắt đầu viết văn bản
    x, y = 60, 10

    # Màu và nội dung văn bản
    text = "{} ".format(class_label[2:])
    text_color = (0, 255, 0)  # Màu văn bản (đen)

    # Sử dụng đối tượng ImageDraw để viết kết quả nhận diện lên hình ảnh
    draw.text((x, y), text, fill=text_color, font=font)

    # Tạo đường dẫn đến tệp tin lưu ảnh mới
    output_path = os.path.join(output_folder, filename)
    # Lưu ảnh mới vào output_folder
    image.save(output_path)

for filename in os.listdir(image_folder):
    # Đường dẫn đầy đủ đến ảnh gốc
    image_path = os.path.join(image_folder, filename)

    Xuly(image_path, output_folder)

# Release OpenCV window and close it
cv2.destroyAllWindows()