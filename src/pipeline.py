from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import cv2
from features import *
from clustering import KMEANS, KMEDOIDS, show_metric
from utils import *
from constant import PATH_OUTPUT, MODEL_CLUSTERING

def load_images_and_labels(main_folder):
    images = []
    labels = []
    file_paths = []  # Pour sauvegarder le chemin de chaque image
    class_names = sorted(os.listdir(main_folder))
    
    for idx_class, class_name in enumerate(class_names):
        class_folder = os.path.join(main_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(class_folder, filename)
                img = cv2.imread(file_path)
                if img is not None:
                    # Pour les features classiques, on redimensionne à 128x128
                    img_resized = cv2.resize(img, (128, 128))
                    images.append(img_resized)
                    labels.append(idx_class)
                    file_paths.append(file_path)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names, file_paths

def pipeline():
    images, labels_true, class_names, file_paths = load_images_and_labels("images/testRegroupe")
    print("Nombre d'images chargées :", len(images))
    print("Labels disponibles :", np.unique(labels_true))
    print("Correspondance ID -> Classe :", dict(enumerate(class_names)))

    print("\n\n ##### Extraction de Features ######")
    print("- calcul features hog...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images)
    print("- calcul features HSV...")
    descriptors_hsv = compute_hsv_histograms(images)
    print("- calcul features SIFT...")
    descriptors_sift = compute_sift_descriptors(images)
    
    # Branche CNN VGG16 : redimensionnement à 224x224
    print("- calcul features CNN (VGG16)...")
    cnn_images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            img_resized = cv2.resize(img, (224, 224))
            cnn_images.append(img_resized)
    cnn_images = np.array(cnn_images)
    descriptors_cnn = compute_cnn_features(cnn_images)
    
    # Branche ResNet50 : redimensionnement à 224x224
    print("- calcul features ResNet50...")
    resnet_images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            img_resized = cv2.resize(img, (224, 224))
            resnet_images.append(img_resized)
    resnet_images = np.array(resnet_images)
    descriptors_resnet = compute_resnet50_features(resnet_images)

    print("\n\n ##### Clustering ######")
    number_cluster = 20  # Correspond aux 20 classes attendues
    if MODEL_CLUSTERING == "kmeans":
        clustered_hog    = KMEANS(number_cluster)
        clustered_hist   = KMEANS(number_cluster)
        clustered_hsv    = KMEANS(number_cluster)
        clustered_sift   = KMEANS(number_cluster)
        clustered_cnn    = KMEANS(number_cluster)
        clustered_resnet = KMEANS(number_cluster)
    elif MODEL_CLUSTERING == "kmedoids":
        clustered_hog    = KMEDOIDS(number_cluster)
        clustered_hist   = KMEDOIDS(number_cluster)
        clustered_hsv    = KMEDOIDS(number_cluster)
        clustered_sift   = KMEDOIDS(number_cluster)
        clustered_cnn    = KMEDOIDS(number_cluster)
        clustered_resnet = KMEDOIDS(number_cluster)
    else:
        raise ValueError("Modèle de clustering non supporté")

    print("- calcul " + MODEL_CLUSTERING + " avec features HOG ...")
    clustered_hog.fit(np.array(descriptors_hog))
    print("- calcul " + MODEL_CLUSTERING + " avec features Histogram...")
    clustered_hist.fit(np.array(descriptors_hist))
    print("- calcul " + MODEL_CLUSTERING + " avec features HSV...")
    clustered_hsv.fit(np.array(descriptors_hsv))
    print("- calcul " + MODEL_CLUSTERING + " avec features SIFT...")
    clustered_sift.fit(np.array(descriptors_sift))
    print("- calcul " + MODEL_CLUSTERING + " avec features CNN (VGG16)...")
    clustered_cnn.fit(np.array(descriptors_cnn))
    print("- calcul " + MODEL_CLUSTERING + " avec features ResNet50...")
    clustered_resnet.fit(np.array(descriptors_resnet))

    print("\n\n ##### Résultat ######")
    metric_hist   = show_metric(labels_true, clustered_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HIST", bool_return=True)
    metric_hog    = show_metric(labels_true, clustered_hog.labels_, descriptors_hog, bool_show=True, name_descriptor="HOG", bool_return=True)
    metric_hsv    = show_metric(labels_true, clustered_hsv.labels_, descriptors_hsv, bool_show=True, name_descriptor="HSV", bool_return=True)
    metric_sift   = show_metric(labels_true, clustered_sift.labels_, descriptors_sift, bool_show=True, name_descriptor="SIFT", bool_return=True)
    metric_cnn    = show_metric(labels_true, clustered_cnn.labels_, descriptors_cnn, bool_show=True, name_descriptor="CNN", bool_return=True)
    metric_resnet = show_metric(labels_true, clustered_resnet.labels_, descriptors_resnet, bool_show=True, name_descriptor="RESNET", bool_return=True)

    print("- export des données vers le dashboard")
    list_dict = [metric_hist, metric_hog, metric_hsv, metric_sift, metric_cnn, metric_resnet]
    df_metric = pd.DataFrame(list_dict)
    
    scaler = StandardScaler()
    descriptors_hist_norm   = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm    = scaler.fit_transform(descriptors_hog)
    descriptors_hsv_norm    = scaler.fit_transform(descriptors_hsv)
    descriptors_sift_norm   = scaler.fit_transform(descriptors_sift)
    descriptors_cnn_norm    = scaler.fit_transform(descriptors_cnn)
    descriptors_resnet_norm = scaler.fit_transform(descriptors_resnet)
    
    x_3d_hist   = conversion_3d(descriptors_hist_norm)
    x_3d_hog    = conversion_3d(descriptors_hog_norm)
    x_3d_hsv    = conversion_3d(descriptors_hsv_norm)
    x_3d_sift   = conversion_3d(descriptors_sift_norm)
    x_3d_cnn    = conversion_3d(descriptors_cnn_norm)
    x_3d_resnet = conversion_3d(descriptors_resnet_norm)

    df_hist   = create_df_to_export(x_3d_hist, labels_true, clustered_hist.labels_)
    df_hog    = create_df_to_export(x_3d_hog, labels_true, clustered_hog.labels_)
    df_hsv    = create_df_to_export(x_3d_hsv, labels_true, clustered_hsv.labels_)
    df_sift   = create_df_to_export(x_3d_sift, labels_true, clustered_sift.labels_)
    df_cnn    = create_df_to_export(x_3d_cnn, labels_true, clustered_cnn.labels_)
    df_resnet = create_df_to_export(x_3d_resnet, labels_true, clustered_resnet.labels_)

    # Génération des mappings image-cluster par descripteur
    mapping_files = {
        "HIST":   (clustered_hist.labels_, "image_clusters_hist.csv"),
        "HOG":    (clustered_hog.labels_, "image_clusters_hog.csv"),
        "HSV":    (clustered_hsv.labels_, "image_clusters_hsv.csv"),
        "SIFT":   (clustered_sift.labels_, "image_clusters_sift.csv"),
        "CNN":    (clustered_cnn.labels_, "image_clusters_cnn.csv"),
        "RESNET": (clustered_resnet.labels_, "image_clusters_resnet.csv")
    }
    
    # Sauvegarder un mapping pour chaque descripteur
    for desc, (labels_pred, file_name) in mapping_files.items():
        df_map = pd.DataFrame({
            "image_path": file_paths,
            "predicted_cluster": labels_pred
        })
        df_map.to_csv(os.path.join(PATH_OUTPUT, file_name), index=False)

    # Sauvegarde des autres résultats
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)
    df_hist.to_excel(os.path.join(PATH_OUTPUT, "save_clustering_hist.xlsx"), index=False, engine='openpyxl')
    df_hog.to_excel(os.path.join(PATH_OUTPUT, "save_clustering_hog.xlsx"), index=False, engine='openpyxl')
    df_hsv.to_excel(os.path.join(PATH_OUTPUT, "save_clustering_hsv.xlsx"), index=False, engine='openpyxl')
    df_sift.to_excel(os.path.join(PATH_OUTPUT, "save_clustering_sift.xlsx"), index=False, engine='openpyxl')
    df_cnn.to_excel(os.path.join(PATH_OUTPUT, "save_clustering_cnn.xlsx"), index=False, engine='openpyxl')
    df_resnet.to_excel(os.path.join(PATH_OUTPUT, "save_clustering_resnet.xlsx"), index=False, engine='openpyxl')
    df_metric.to_excel(os.path.join(PATH_OUTPUT, "save_metric.xlsx"), index=False, engine='openpyxl')

    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard.py")

if __name__ == "__main__":
    pipeline()
