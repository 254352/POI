import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd


def calculate_texture_features(input_dir, output_file):
    # Lista nazw cech
    feature_names = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

    # Przyjęte odległości pikseli i kierunki
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 stopni

    # Lista cech
    all_features = []

    # Iteracja przez pliki w katalogu
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)

            # Wczytaj obraz, przekształć do skali szarości i zmniejsz głębię jasności do 5 bitów
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = np.uint8(image // 64)

            # Oblicz macierz zdarzeń
            glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

            # Oblicz cechy tekstury
            features = []
            for feature_name in feature_names:
                feature_values = graycoprops(glcm, feature_name)
                features.extend(feature_values.flatten())

            # Dodaj nazwę kategorii tekstury
            features.append(category)

            # Dodaj cechy do listy
            all_features.append(features)

    # Zapisz cechy do pliku CSV
    columns = [f'{name}_{distance}_{angle}' for name in feature_names for distance in distances for angle in angles]
    columns.append('category')
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(output_file, index=False)


# Przykładowe użycie
input_directory = 'probki'
output_file = 'wyniki_cech.csv'

calculate_texture_features(input_directory, output_file)
