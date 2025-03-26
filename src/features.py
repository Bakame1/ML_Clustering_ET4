from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools

def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images en couleur.
    Input : images (list) : liste des images en couleur
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    for image in images:
        # Calcul de l'histogramme pour chaque canal de couleur
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

        # Normalisation des histogrammes
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()

        # Concaténation des histogrammes des trois canaux
        hist = np.concatenate((hist_r, hist_g, hist_b))
        descriptors.append(hist)

    return descriptors

def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en couleur.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    for image in images:
        fd = hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False, channel_axis=-1)
        descriptors.append(fd)
    return descriptors

def compute_hsv_histograms(images):
    """
    Calcule les histogrammes HSV pour les images en couleur.
    Input : images (list) : liste des images en couleur
    Output : descriptors (list) : liste des descripteurs d'histogrammes HSV
    """
    descriptors = []
    for image in images:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        descriptors.append(hist)
    return descriptors

def compute_sift_descriptors(images, num_descriptors=10):
    """
    Calcule les descripteurs SIFT pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs SIFT
    """
    sift = cv2.SIFT_create()
    descriptors = []
    for image in images:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, desc = sift.detectAndCompute(image, None)
        if desc is not None:
            if len(desc) > num_descriptors:
                desc = desc[:num_descriptors]
            else:
                desc = np.vstack((desc, np.zeros((num_descriptors - len(desc), 128))))
            descriptors.append(desc.flatten())
        else:
            descriptors.append(np.zeros(num_descriptors * 128))
    return descriptors

def compute_cnn_features(images):
    """
    Calcule les features CNN pour les images en couleur.
    Les images doivent être de taille 224x224.
    On utilise VGG16 pré-entraîné (sans la dernière couche de classification) et un pooling global.
    Input : images (array) : tableau numpy des images (taille 224x224)
    Output : descriptors (array) : tableau de features extraites
    """
    

    # Prétraitement
    images = images.astype('float32')
    images = preprocess_input(images)
    
    # Charger VGG16 sans les couches de classification et avec un pooling global
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    descriptors = base_model.predict(images)
    return descriptors
