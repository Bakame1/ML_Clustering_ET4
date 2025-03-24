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
        # Calcul des descripteurs HOG
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
        # Conversion de l'image en couleur en HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calcul de l'histogramme HSV
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalisation de l'histogramme

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
        # Conversion en uint8
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calcul des descripteurs SIFT
        _, desc = sift.detectAndCompute(image, None)

        if desc is not None:
            # Limiter le nombre de descripteurs à num_descriptors
            if len(desc) > num_descriptors:
                desc = desc[:num_descriptors]
            else:
                # Remplir avec des zéros si moins de descripteurs
                desc = np.vstack((desc, np.zeros((num_descriptors - len(desc), 128))))

            descriptors.append(desc.flatten())
        else:
            # Si aucun descripteur n'est trouvé, ajouter une liste vide ou un vecteur de zéros
            descriptors.append(np.zeros(num_descriptors * 128))  # SIFT a généralement 128 dimensions

    return descriptors
