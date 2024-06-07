# This file is used to obtain the vector of people features
# locally, using the pre-trained FaceNet network.

import os
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import LabelEncoder
import pickle

# Function to extract facial embeddings (vector) using FaceNet
def extract_face_embeddings(img_path):
    img = Image.open(img_path).convert('RGB')
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        with torch.no_grad():
            embedding = model(img_cropped.unsqueeze(0).to(device))
        return embedding.squeeze().cpu().numpy()
    return None

# Settings
if torch.cuda.is_available():
    print('using cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN model for face detection
mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

# FaceNet model (InceptionResnetV1)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Lists to store embeddings and labels
embeddings_list = []
labels_list = []

data_dir = 'people'

# Iterate over each person's folders
for person_folder in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_folder)
    if os.path.isdir(person_path):
        # Iterate over the current person's images
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            
            # Extract the facial embedding from the current image
            embedding = extract_face_embeddings(img_path)
            if embedding is not None:
                embeddings_list.append(embedding)
                labels_list.append(person_folder)

# Convert the lists to numpy arrays
embeddings = np.array(embeddings_list)
labels = np.array(labels_list)

# Encode the labels into numbers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save the embeddings and labels into numpy files
np.save('embeddings.npy', embeddings)
np.save('labels.npy', encoded_labels)

# Save the LabelEncoder into a file
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
