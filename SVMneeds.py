import os
import cv2
import dlib
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
from matplotlib import pyplot as plt
from openpyxl import load_workbook
from pathlib import Path
import streamlit as st

class ImagePreprocessor:
    def __init__(self, image):
        self.image = image
        self.resized_image = None
        self.rotated_image = None

    def read_and_resize(self, scale_percent=100):
        try:
            if len(self.image.shape) == 2:  # Grayscale
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
            elif self.image.shape[2] == 4:  # RGBA
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)
            elif self.image.shape[2] != 3:
                raise ValueError(f"Unsupported image shape: {self.image.shape}")

            # Log original image properties
            print(f"Original image shape: {self.image.shape}, dtype: {self.image.dtype}")

            width = int(self.image.shape[1] * scale_percent / 100)
            height = int(self.image.shape[0] * scale_percent / 100)
            dim = (width, height)
            self.resized_image = cv2.resize(self.image, dim)

            # Log resized image properties
            print(f"Resized image shape: {self.resized_image.shape}, dtype: {self.resized_image.dtype}")

            if self.resized_image.dtype != 'uint8':
                self.resized_image = self.resized_image.astype('uint8')
        except Exception as e:
            print(f"Error in read_and_resize: {e}")
            raise e

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
        plt.show()

class LandmarkExtractor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    def extract_landmarks(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            st.write(f"Grayscale image shape: {gray.shape}, dtype: {gray.dtype}")
            
            faces = self.detector(gray)
            st.write(f"Number of faces detected: {len(faces)}")

            if len(faces) == 0:
                return None

            for face in faces:
                landmarks = self.predictor(image=gray, box=face)
                return np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
        except Exception as e:
            st.error(f"Error in extract_landmarks: {e}")
        return None
    
    def visualize_landmarks(self, image, landmarks):
        if landmarks is None:
            print("No landmarks to visualize.")
            return
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=20, marker='.', c='c')
        plt.show()
        
class FeatureCalculator:
    
    #Rumus Angle
    @staticmethod
    def Rumus1(rumus,xa1,xa2,ya1,ya2):
        Deltax = abs(xa1-xa2)
        Deltay = abs(ya1-ya2)
        if rumus == 'sudut': #Rumus Sudut
            theta = np.arctan2(Deltay, Deltax)
            sudut = theta*(180/np.pi)
            return sudut
        elif rumus == 'slope': #Rumus Slope
            slope = abs(Deltay/Deltax)
            return slope
        elif rumus == 'jarak': #Rumus Jarak
            jarak = np.sqrt((Deltax)**2 + (Deltay)**2)
            return jarak

    #Rumus Maximum
    @staticmethod
    def Maximum(Nilai1, Nilai2):
        max = np.maximum(Nilai1/Nilai2,Nilai2/Nilai1)
        return max
        
    #Rumus Average
    @staticmethod
    def avg(yv1,yv2,yv3,yv4,yv5):
        return ((yv1+yv2+yv3+yv4+yv5)/5)

    #Rumus Gabungan
    @staticmethod
    def gabungan(landmarks):
        pl =[landmarks[48],landmarks[49],landmarks[50],landmarks[51],landmarks[57],landmarks[58],landmarks[59]]
        pr =[landmarks[51],landmarks[52],landmarks[53],landmarks[54],landmarks[55],landmarks[56],landmarks[57]]
        
        sum = 0
        for i in range(len(pl)-1):
            f1 = FeatureCalculator.Rumus1('jarak',pl[i][0], pl[i+1][0], pl[i][1],pl[i+1][1])
            sum = sum + f1
        result1 = sum + FeatureCalculator.Rumus1('jarak',landmarks[48,0],landmarks[59,0],landmarks[48,1],landmarks[59,1])
        sum = 0
        for i in range(len(pr)-1):
            f2 = FeatureCalculator.Rumus1('jarak',pr[i][0], pr[i+1][0], pr[i][1],pr[i+1][1])
            sum = sum + f2
        result2 = sum + FeatureCalculator.Rumus1('jarak',landmarks[51,0],landmarks[57,0],landmarks[51,1],landmarks[57,1])
        return result1, result2

    @staticmethod
    def rumus29 (landmarks):
        #f0.1
        f0_1 = FeatureCalculator.Rumus1('sudut',landmarks[17,0],landmarks[26,0],landmarks[17,1],landmarks[26,1])
        
        #f1.1
        f1_1 = FeatureCalculator.Rumus1('sudut',landmarks[19,0],landmarks[24,0],landmarks[19,1],landmarks[24,1])

        #f2.1
        f2_1 = FeatureCalculator.Rumus1('sudut',landmarks[21,0],landmarks[22,0],landmarks[21,1],landmarks[22,1])
        
        #f3.1
        L = FeatureCalculator.avg(landmarks[17,1],landmarks[18,1],landmarks[19,1],landmarks[20,1],landmarks[21,1])
        M = FeatureCalculator.avg(landmarks[22,1],landmarks[23,1],landmarks[24,1],landmarks[25,1],landmarks[26,1])
        f3_1 = FeatureCalculator.Maximum(L,M)

        #f4.1
        f4_1 = FeatureCalculator.Rumus1('slope',landmarks[17,0],landmarks[26,0],landmarks[17,1],landmarks[26,1])

        #f5.1
        f5_1 = FeatureCalculator.Rumus1('slope',landmarks[19,0],landmarks[24,0],landmarks[19,1],landmarks[24,1])

        #f6.1
        f6_1 = FeatureCalculator.Rumus1('slope',landmarks[21,0],landmarks[22,0],landmarks[21,1],landmarks[22,1])

        #f7.1
        f7_1 = FeatureCalculator.Rumus1('sudut',landmarks[36,0],landmarks[45,0],landmarks[36,1],landmarks[45,1])

        #f10.1
        H = FeatureCalculator.Rumus1('jarak',landmarks[31,0],landmarks[36,0],landmarks[31,1],landmarks[36,1])
        I = FeatureCalculator.Rumus1('jarak',landmarks[35,0],landmarks[45,0],landmarks[35,1],landmarks[45,1])
        f10_1 = FeatureCalculator.Maximum(H,I)

        #f3.2
        L = FeatureCalculator.avg(landmarks[17,1],landmarks[18,1],landmarks[19,1],landmarks[20,1],landmarks[21,1])
        M = FeatureCalculator.avg(landmarks[22,1],landmarks[23,1],landmarks[24,1],landmarks[25,1],landmarks[26,1])
        f3_2 = FeatureCalculator.Maximum(L,M)

        #f8.2
        Bl = FeatureCalculator.Rumus1('jarak',landmarks[36,0],landmarks[39,0],landmarks[36,1],landmarks[39,1])
        Br = FeatureCalculator.Rumus1('jarak',landmarks[42,0],landmarks[45,0],landmarks[42,1],landmarks[45,1])
        f8_2 = FeatureCalculator.Maximum(Bl,Br)

        #f9.2
        D = FeatureCalculator.Rumus1('jarak',landmarks[0,0],landmarks[36,0],landmarks[0,1],landmarks[36,1])
        E = FeatureCalculator.Rumus1('jarak',landmarks[45,0],landmarks[16,0],landmarks[45,1],landmarks[16,1])
        f9_2 = FeatureCalculator.Maximum(D,E)

        #f10.2
        H = FeatureCalculator.Rumus1('jarak',landmarks[31,0],landmarks[36,0],landmarks[31,1],landmarks[36,1])
        I = FeatureCalculator.Rumus1('jarak',landmarks[35,0],landmarks[45,0],landmarks[35,1],landmarks[45,1])
        f10_2 = FeatureCalculator.Maximum(H,I)

        #f15.2
        F = FeatureCalculator.Rumus1('jarak',landmarks[57,0],landmarks[36,0],landmarks[57,1],landmarks[36,1])
        G = FeatureCalculator.Rumus1('jarak',landmarks[57,0],landmarks[45,0],landmarks[57,1],landmarks[45,1])
        f15_2 = FeatureCalculator.Maximum(F,G)

        #f27.2
        A = FeatureCalculator.Rumus1('jarak',landmarks[0,0],landmarks[16,0],landmarks[0,1],landmarks[16,1])
        C = FeatureCalculator.Rumus1('jarak',landmarks[8,0],landmarks[57,0],landmarks[8,1],landmarks[57,1])
        f27_2 = (C/A)

        #f1.3
        f1_3 = FeatureCalculator.Rumus1('sudut',landmarks[19,0],landmarks[24,0],landmarks[19,1],landmarks[24,1])

        #f3.3
        L = FeatureCalculator.avg(landmarks[17,1],landmarks[18,1],landmarks[19,1],landmarks[20,1],landmarks[21,1])
        M = FeatureCalculator.avg(landmarks[22,1],landmarks[23,1],landmarks[24,1],landmarks[25,1],landmarks[26,1])
        f3_3 = FeatureCalculator.Maximum(L,M)

        #f8.3
        Bl = FeatureCalculator.Rumus1('jarak',landmarks[36,0],landmarks[39,0],landmarks[36,1],landmarks[39,1])
        Br = FeatureCalculator.Rumus1('jarak',landmarks[42,0],landmarks[45,0],landmarks[42,1],landmarks[45,1])
        f8_3 = FeatureCalculator.Maximum(Bl,Br)

        #f9.3
        D = FeatureCalculator.Rumus1('jarak',landmarks[0,0],landmarks[36,0],landmarks[0,1],landmarks[36,1])
        E = FeatureCalculator.Rumus1('jarak',landmarks[45,0],landmarks[16,0],landmarks[45,1],landmarks[16,1])
        f9_3 = FeatureCalculator.Maximum(D,E)

        #f10.3
        H = FeatureCalculator.Rumus1('jarak',landmarks[31,0],landmarks[36,0],landmarks[31,1],landmarks[36,1])
        I = FeatureCalculator.Rumus1('jarak',landmarks[35,0],landmarks[45,0],landmarks[35,1],landmarks[45,1])
        f10_3 = FeatureCalculator.Maximum(H,I)

        #f15.3
        F = FeatureCalculator.Rumus1('jarak',landmarks[57,0],landmarks[36,0],landmarks[57,1],landmarks[36,1])
        G = FeatureCalculator.Rumus1('jarak',landmarks[57,0],landmarks[45,0],landmarks[57,1],landmarks[45,1])
        f15_3 = FeatureCalculator.Maximum(F,G)

        #f18.3
        Vl = FeatureCalculator.Rumus1('jarak',landmarks[57,0],landmarks[48,0],landmarks[57,1],landmarks[48,1])
        Vr = FeatureCalculator.Rumus1('jarak',landmarks[57,0],landmarks[54,0],landmarks[57,1],landmarks[54,1])
        f18_3 = np.maximum(Vl/A,Vr/A)

        #f21.3
        W = FeatureCalculator.Rumus1('jarak',landmarks[48,0],landmarks[54,0],landmarks[48,1],landmarks[54,1])
        Wl, Wr = FeatureCalculator.gabungan(landmarks)
        f21_3 = np.maximum(Wl/W,Wr/W)

        #f23.3
        f23_3 = FeatureCalculator.Rumus1('sudut',landmarks[30,0],landmarks[57,0],landmarks[30,1],landmarks[57,1])

        #f28.3
        X = FeatureCalculator.Rumus1('jarak',landmarks[33,0],landmarks[51,0],landmarks[33,1],landmarks[51,1])
        f28_3 = (X/A)

        return (f0_1, f1_1, f2_1, f3_1, f4_1, f5_1, f6_1, f7_1, f10_1, f3_2, f8_2, f9_2, f10_2, f15_2, f27_2, f1_3, f3_3, f8_3, f9_3, f10_3, f15_3, f18_3, f21_3, f23_3, f28_3)

class ImageSlopeCorrector:
    
    @staticmethod
    def rotate_image_based_on_landmarks(image, landmarks):
        kemiringan = FeatureCalculator.Rumus1('sudut',landmarks[0,0],landmarks[16,0],landmarks[0,1],landmarks[16,1])
        print('Dirotasi sebesar: '+ str(kemiringan) +str(' derajat'))
        fixed_image = ndi.rotate(image, angle=-kemiringan, reshape=False)
        fixed_image = fixed_image.astype('uint8')
        return fixed_image

    @staticmethod
    def show_image(image):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

class DataPersistence:
    def __init__(self, output_file):
        self.output_file = output_file

    def save_to_excel(self, data, sheet_name):
        # Check if the Excel file already exists to decide on the mode
        try:
            with pd.ExcelWriter(self.output_file, mode='a', engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name=sheet_name)
        except FileNotFoundError:
            # File does not exist, will create a new one
            data.to_excel(self.output_file, sheet_name=sheet_name)

    def create_dataframe(self, features, image_name):
        # Assuming 'features' is a dictionary with feature names as keys and calculated values as values
        df = pd.DataFrame([features], index=[image_name])
        return df

