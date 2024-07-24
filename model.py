import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import efficientnet.tfkeras as efn
import os

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Tải mô hình EfficientNet B0
base_model = efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = tf.keras.applications.efficientnet.preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Đường dẫn tới thư mục chứa ảnh
image_dir = './data/train_preprocessed'
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.jpeg'))]

# Trích xuất đặc trưng cho tất cả ảnh
features = np.array([extract_features(img_path, base_model) for img_path in image_paths])

# Tìm ảnh giống nhất
def find_most_similar(image_features, features):
    similarities = cosine_similarity([image_features], features)
    most_similar_idx = np.argmax(similarities)
    return image_paths[most_similar_idx]

# Đường dẫn tới ảnh cần tìm kiếm
query_image_path = './data/test_preprocessed/image_49.jpg'
query_features = extract_features(query_image_path, base_model)
most_similar_image = find_most_similar(query_features, features)

print("Most similar image:", most_similar_image)
