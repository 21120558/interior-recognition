import os
import re
import cv2
import numpy as np
import random
import shutil

# Đường dẫn cơ bản
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, 'image')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Hàm xử lý ảnh với padding
def preprocess_image_with_padding(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    old_size = img.shape[:2]

    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    new_img = np.full((target_size[0], target_size[1], 3), 255, dtype=np.uint8)
    new_img[(target_size[0] - new_size[0]) // 2:(target_size[0] - new_size[0]) // 2 + new_size[0],
            (target_size[1] - new_size[1]) // 2:(target_size[1] - new_size[1]) // 2 + new_size[1]] = img

    return new_img

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Tạo các thư mục con cho từng lớp trong train và test
classes = ['L', 'H', 'U', 'I', 'T']
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

# Tỷ lệ chia dữ liệu train và test
train_ratio = 1

# Hàm di chuyển tệp hình ảnh vào thư mục đích
def move_images(filename, class_name, src_dir, train_dest_dir, test_dest_dir):
    src_path = os.path.join(src_dir, filename)
    if os.path.exists(src_path):
        img = preprocess_image_with_padding(src_path)

        if filename.lower().endswith('.webp'):
            filename = filename.rsplit('.', 1)[0] + '.jpg'

        if random.random() < train_ratio:
            dst_path = os.path.join(train_dest_dir, class_name, filename)
        else:
            dst_path = os.path.join(test_dest_dir, class_name, filename)
        cv2.imwrite(dst_path, img)
        print(f'Moved {filename} to {dst_path}')

# Duyệt qua các ảnh trong thư mục image và di chuyển vào thư mục tương ứng
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        class_name = filename[0].upper()  # Lấy chữ cái đầu tiên làm lớp
        move_images(filename, class_name, image_dir, train_dir, test_dir)

print('Done!')
