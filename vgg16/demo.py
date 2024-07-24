import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

data_dir = '../data/train'
model_path = 'vgg16_places365_finetuned_final.keras'

model = load_model(model_path)

feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)  # Lớp GlobalAveragePooling2D

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale
    return img_array

input_image_path = '../data/test/U/image_7.png'
input_image = load_and_preprocess_image(input_image_path)
input_feature = feature_extractor.predict(input_image)

# Tải và trích xuất đặc trưng từ toàn bộ ảnh trong tập dữ liệu
image_paths = []
features = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(root, file)
            img = load_and_preprocess_image(img_path)
            feature = feature_extractor.predict(img)
            image_paths.append(img_path)
            features.append(feature)

features = np.vstack(features)

similarities = cosine_similarity(input_feature, features)
most_similar_index = np.argmax(similarities)
most_similar_image_path = image_paths[most_similar_index]

print(f"The most similar image is: {most_similar_image_path}")
