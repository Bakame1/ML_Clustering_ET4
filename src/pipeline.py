from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import cv2
from features import *
from clustering import KMEANS, KMEDOIDS, Agglomerative_Clustering, show_metric
from utils import *
from constant import PATH_OUTPUT

def load_images_and_labels(main_folder):
    images = []
    labels = []
    file_paths = []  #Pour sauvegarder le chemin de chaque image
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
                    #Pour les features classiques, redimensionner à 128x128
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

    print("\n\n##### Extraction de Features #####")
    print("- calcul features HOG...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images)
    print("- calcul features HSV...")
    descriptors_hsv = compute_hsv_histograms(images)
    print("- calcul features SIFT...")
    descriptors_sift = compute_sift_descriptors(images)
    
    #Branche CNN VGG16 : redimensionnement à 224x224
    print("- calcul features CNN (VGG16)...")
    cnn_images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            img_resized = cv2.resize(img, (224, 224))
            cnn_images.append(img_resized)
    cnn_images = np.array(cnn_images)
    descriptors_cnn = compute_cnn_features(cnn_images)
    
    #Branche ResNet50 : redimensionnement à 224x224
    print("- calcul features ResNet50...")
    resnet_images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            img_resized = cv2.resize(img, (224, 224))
            resnet_images.append(img_resized)
    resnet_images = np.array(resnet_images)
    descriptors_resnet = compute_resnet50_features(resnet_images)

    #Dictionnaire des jeux de descripteurs et leur nom
    descriptors_dict = {
        "HIST": descriptors_hist,
        "HOG": descriptors_hog,
        "HSV": descriptors_hsv,
        "SIFT": descriptors_sift,
        "CNN": descriptors_cnn,
        "RESNET": descriptors_resnet
    }

    #Dictionnaire reliant le nom de l'algorithme à sa classe
    clustering_models = {
        "kmeans": KMEANS,
        "kmedoids": KMEDOIDS,
        "agglomerative": Agglomerative_Clustering
    }

    number_cluster = 20  #Nombre de clusters attendu
    metric_results = []  #Pour regrouper les métriques de chaque algorithme

    #Pour chaque algo de clustering
    for algo_name, AlgoClass in clustering_models.items():
        print(f"\n--- Clustering avec {algo_name.upper()} ---")
        #Pour chaque descripteur
        for desc_name, descriptors in descriptors_dict.items():
            print(f"- calcul {algo_name} avec features {desc_name} ...")
            clustering_model = AlgoClass(number_cluster)

            clustering_model.fit(np.array(descriptors))
            #Calcul des métriques
            metrics_dict = show_metric(
                labels_true, clustering_model.labels_, descriptors,
                bool_show=True, name_descriptor=desc_name,name_model=algo_name, bool_return=True
            )
            metrics_dict["algo"] = algo_name
            metric_results.append(metrics_dict)

            #Sauvegarde du mapping image-cluster pour ce descripteur et cet algo
            mapping_file = os.path.join(PATH_OUTPUT, f"image_clusters_{desc_name.lower()}_{algo_name}.csv")
            df_map = pd.DataFrame({
                "image_path": file_paths,
                "predicted_cluster": clustering_model.labels_
            })
            df_map.to_csv(mapping_file, index=False)

            #Sauvegarde des résultats de clustering (en 3D) pour le dashboard
            scaler = StandardScaler()
            descriptors_norm = scaler.fit_transform(descriptors)
            x_3d = conversion_3d(descriptors_norm)
            df_export = create_df_to_export(x_3d, labels_true, clustering_model.labels_)
            file_export = os.path.join(PATH_OUTPUT, f"save_clustering_{desc_name.lower()}_{algo_name}.xlsx")
            #Exportation excel
            df_export.to_excel(file_export, index=False, engine='openpyxl')

    #Sauvegarde globale des métriques pour tous les algos
    df_metric = pd.DataFrame(metric_results)
    df_metric.to_excel(os.path.join(PATH_OUTPUT, "save_metric_all.xlsx"), index=False, engine='openpyxl')

    print("Fin.\n\nPour avoir la visualisation dashboard, lancez la commande : streamlit run dashboard.py")

if __name__ == "__main__":
    pipeline()