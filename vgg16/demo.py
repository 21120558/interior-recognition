import json
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from PIL import Image
import cv2

train_data_dir = '../data/train'
model_path = 'vgg16_places365_finetuned_final.keras'
class_indices_path = 'class_indices.json'


model = load_model(model_path)

feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)  # Lớp GlobalAveragePooling2D

def preprocess_image_with_padding(img_path, target_size=(224, 224), first=False):
    img = cv2.imread(img_path)
    old_size = img.shape[:2]

    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    new_img = np.full((target_size[0], target_size[1], 3), 255, dtype=np.uint8)
    new_img[(target_size[0] - new_size[0]) // 2:(target_size[0] - new_size[0]) // 2 + new_size[0],
            (target_size[1] - new_size[1]) // 2:(target_size[1] - new_size[1]) // 2 + new_size[1]] = img

    if first == True:
        cv2.imwrite('test.png', new_img)

    return new_img


def load_and_preprocess_image(img_path, target_size=(224, 224), first=False):
    preprocessed_img = preprocess_image_with_padding(img_path, target_size, first)
    img = Image.fromarray(preprocessed_img)
    img = img.resize(target_size)  # Ensure it's the correct target size


    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale
    return img_array

with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
class_indices = {int(v): k for k, v in class_indices.items()}

input_image_path = '../data/test/U/image_7.png'

def predict(input_image_path):
    input_image = load_and_preprocess_image(input_image_path, first=True)
    predicted_class = np.argmax(model.predict(input_image))

    predicted_class_name = class_indices[predicted_class]

    input_feature = feature_extractor.predict(input_image)

    image_paths = []
    features = []
    class_dir = os.path.join(train_data_dir, predicted_class_name)

    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('png', 'jpg', 'jpeg', 'webp')) and os.path.join(class_dir, img_name) != input_image_path:
            img_path = os.path.join(class_dir, img_name)
            img = load_and_preprocess_image(img_path, first=False)
            feature = feature_extractor.predict(img)
            image_paths.append(img_path)
            features.append(feature)

    features = np.vstack(features)

    similarities = cosine_similarity(input_feature, features)
    most_similar_indices = np.argsort(similarities[0])[-3:][::-1]
    most_similar_image_paths = [image_paths[i] for i in most_similar_indices]

    for i, path in enumerate(most_similar_image_paths):
        print(f"The {i+1} most similar image in class {predicted_class_name} is: {path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Interior Detection')
    parser.add_argument('--input', type=str, help='Path to the input image')
    args = parser.parse_args()

    input_image_path = args.input

    predict(input_image_path)
