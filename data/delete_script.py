import os
import re
import cv2
import numpy as np
import random
import shutil


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')
file_list = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
pattern = re.compile(r'\(1\)\.')

for filename in file_list:
    if pattern.search(filename):
        os.remove(os.path.join(image_dir, filename))
        print(f"Deleted duplicate file: {filename}")

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

image_pre_dir = os.path.join(base_dir, 'image_preprocessed')

if not os.path.exists(image_pre_dir):
    os.makedirs(image_pre_dir)

for i, filename in enumerate(os.listdir(image_dir), start=1):
    if filename.endswith(('.jpg', '.webp', '.png', '.jpeg')):
        input_path = os.path.join(image_dir, filename)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(image_pre_dir, f'image_{i}{ext}')
        img_data = preprocess_image_with_padding(input_path)
        cv2.imwrite(output_path, img_data)
        print(f"Processed {input_path} -> {output_path}")

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = ['L', 'H', 'U', 'I', 'T']
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

L = [1, 10, 15, 16, 21, 24, 26, 27, 28, 32, 34, 35, 36, 37, 39, 40, 41, 42, 44, 46, 49, 50, 51, 52, 55, 57, 58, 60, 61, 63, 66, 67, 68, 72, 73, 75, 76, 80, 81, 82, 83, 85, 91, 95, 101, 102, 103, 105, 111, 112, 114, 115, 116, 117, 118, 120, 121, 123, 124, 125, 126]
H = [3, 5, 6, 13, 17, 31, 64, 70, 71, 77, 78, 100, 104, 106, 107, 109, 119]
U = [2, 7, 8, 18, 19, 22, 23, 33, 43, 45, 47, 48, 53, 56, 79, 86, 87, 89, 92, 93, 113, 122]
I = [4, 9, 11, 14, 20, 25, 29, 30, 38, 54, 59, 62, 65, 69, 74, 84, 90, 94, 96, 97, 98, 99, 108, 110]
T = [12, 88]

# Dictionary để map các lớp với các hình ảnh tương ứng
image_classes = {
    'L': L,
    'H': H,
    'U': U,
    'I': I,
    'T': T,
}

# Tỷ lệ chia dữ liệu train và test
train_ratio = 0.9

# Hàm để di chuyển tệp hình ảnh vào thư mục đích
def move_images(image_numbers, class_name, src_dir, train_dest_dir, test_dest_dir):
    for img_num in image_numbers:
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            img_name = f'image_{img_num}{ext}'
            src_path = os.path.join(src_dir, img_name)
            if os.path.exists(src_path):
                if random.random() < train_ratio:
                    dst_path = os.path.join(train_dest_dir, class_name, img_name)
                else:
                    dst_path = os.path.join(test_dest_dir, class_name, img_name)
                shutil.move(src_path, dst_path)
                print(f'Moved {img_name} to {dst_path}')

for cls, images in image_classes.items():
    print(train_dir)
    move_images(images, cls, image_pre_dir, train_dir, test_dir)

print('Done!')