import os
import shutil
import random

train_ratio = 0.8

directory = os.path.dirname(os.path.abspath(__file__))

valid_extensions = {'.png', '.jpg', '.webp', '.jpeg'}
file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in valid_extensions]

renamed_files = []
for i, filename in enumerate(file_list, start=1):
    name, ext = os.path.splitext(filename)
    new_name = f'image_{i}{ext}'
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
    renamed_files.append(new_name)

os.makedirs(os.path.join(directory, 'train'), exist_ok=True)
os.makedirs(os.path.join(directory, 'test'), exist_ok=True)
train_path = os.path.join(directory, 'train')
test_path = os.path.join(directory, 'test')

random.shuffle(renamed_files)
split_index = int(len(renamed_files) * train_ratio)
train_files = renamed_files[:split_index]
test_files = renamed_files[split_index:]

for f in train_files:
    shutil.move(os.path.join(directory, f), os.path.join(train_path, f))

for f in test_files:
    shutil.move(os.path.join(directory, f), os.path.join(test_path, f))