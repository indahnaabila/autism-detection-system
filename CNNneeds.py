import cv2
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import dlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

# Function to extract frames
def extract_frames(video_path, output_folder, desired_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(video_fps / desired_fps)

    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        frame_count += 1
    
    cap.release()
    return extracted_count

# Class for image preprocessing
class ImagePreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.resized_image = None
        self.rotated_image = None
    
    def read_and_resize(self, scale_percent=100):
        img = cv2.imread(self.image_path)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        self.resized_image = cv2.resize(img, dim)
    
    def rotate(self, clockwise):
        if self.resized_image is None:
            print("Image needs to be resized first.")
            return
        if clockwise:
            self.rotated_image = cv2.rotate(self.resized_image, cv2.ROTATE_90_CLOCKWISE)
        else:
            self.rotated_image = cv2.rotate(self.resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def show_image(self, image):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Class for extracting landmarks
class LandmarkExtractor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    def extract_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        for face in faces:
            landmarks = self.predictor(image=gray, box=face)
            return np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
        return None
    
    def visualize_landmarks(self, landmarks, save_path=None, output_size=(224, 224)):
        if landmarks is None:
            print("No landmarks to visualize.")
            return
        fig = plt.figure(figsize=(8, 8), facecolor='white')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=20, marker='o', c='black')
        plt.gca().invert_yaxis()  # Invert y-axis to match the image coordinate system
        plt.axis('equal')  # Equal scaling for both axes
        plt.axis('off')  # Turn off the axis
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white')
            plt.close()

            # Resize the saved image to the required output size
            try:
                image = cv2.imread(save_path)
                if image is not None:
                    resized_image = cv2.resize(image, output_size)
                    cv2.imwrite(save_path, resized_image)
                else:
                    print(f"Error reading image: {save_path}")
            except Exception as e:
                print(f"Error processing image: {save_path}, {str(e)}")
        else:
            plt.show()

class FeatureCalculator:
    @staticmethod
    def Rumus1(rumus, xa1, xa2, ya1, ya2):
        Deltax = abs(xa1 - xa2)
        Deltay = abs(ya1 - ya2)
        if rumus == 'sudut': # Rumus Sudut
            theta = np.arctan2(Deltay, Deltax)
            sudut = theta * (180 / np.pi)
            return sudut
        elif rumus == 'slope': # Rumus Slope
            slope = abs(Deltay / Deltax)
            return slope
        elif rumus == 'jarak': # Rumus Jarak
            jarak = np.sqrt((Deltax) ** 2 + (Deltay) ** 2)
            return jarak

    @staticmethod
    def Maximum(Nilai1, Nilai2):
        max = np.maximum(Nilai1 / Nilai2, Nilai2 / Nilai1)
        return max

    @staticmethod
    def avg(yv1, yv2, yv3, yv4, yv5):
        return ((yv1 + yv2 + yv3 + yv4 + yv5) / 5)



class ImageSlopeCorrector:
    @staticmethod
    def rotate_image_based_on_landmarks(image, landmarks):
        kemiringan = FeatureCalculator.Rumus1('sudut', landmarks[0, 0], landmarks[16, 0], landmarks[0, 1], landmarks[16, 1])
        print('Dirotasi sebesar: ' + str(kemiringan) + str(' derajat'))
        fixed_image = ndi.rotate(image, angle=-kemiringan, reshape=False)
        return fixed_image

    @staticmethod
    def show_image(image):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


# Function to process images in a folder
def process_images_in_folder(folder_path, output_folder, output_size=(224, 224), batch_size=10):
    extractor = LandmarkExtractor()
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                all_files.append(os.path.join(root, file))
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        for file_path in batch_files:
            preprocessor = ImagePreprocessor(file_path)
            preprocessor.read_and_resize()
            resize_image = preprocessor.resized_image
            landmarks = extractor.extract_landmarks(resize_image)
            if landmarks is not None:
                corrected_image = ImageSlopeCorrector.rotate_image_based_on_landmarks(resize_image, landmarks)
                corrected_landmarks = extractor.extract_landmarks(corrected_image)
                relative_path = os.path.relpath(os.path.dirname(file_path), folder_path)
                save_dir = os.path.join(output_folder, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_landmarks.png")
                extractor.visualize_landmarks(corrected_landmarks, save_path=save_path, output_size=output_size)

# Function to analyze data
def analyze_data_np(data):
    if len(data) == 0:
        return None, None, None

    count_0 = np.count_nonzero(data == 0)
    count_1 = np.count_nonzero(data)

    # Calculate probabilities
    probability_0 = count_0 / len(data)
    probability_1 = count_1 / len(data)

    # Make a decision based on the probabilities
    decision = "Class 1" if probability_1 > probability_0 else "Class 0"

    return probability_0, probability_1, decision

# Define the ImprovedCNN class (make sure this matches the one used for training)
class ImprovedCNN(nn.Module):
    def __init__(self, kernel_size, filters, dropout_rate):
        super(ImprovedCNN, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(3, filters, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters * 2, filters * 4, kernel_size=kernel_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(filters * 4, filters * 8, kernel_size=kernel_size, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.flattened_size = self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 1)
        
    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # Assuming input images are 224x224
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            return x.numel()
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Specify the best parameters used during training
best_params = {'kernel_size': 4, 'filters': 32, 'dropout_rate': 0.32066245874503196, 'learning_rate': 0.00014398189969825852}

# Initialize the model
model = ImprovedCNN(best_params['kernel_size'], best_params['filters'], best_params['dropout_rate'])

# Load the model state using the custom name with map_location to CPU
custom_model_name = "CNN_N4_100w.pth"
model.load_state_dict(torch.load(custom_model_name, map_location=torch.device('cpu')))

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to predict the class of an input image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.sigmoid(outputs).item()
        predicted_class = 1 if predicted > 0.5 else 0
    return predicted_class

# Function to predict all images in a folder
def predict_folder(folder_path):
    predictions = []
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            predicted_class = predict_image(image_path)
            predictions.append(predicted_class)
            print(f'Frame {i + 1} predicted as {predicted_class}')
    return predictions
