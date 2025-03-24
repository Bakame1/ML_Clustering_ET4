from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from sklearn import datasets

from features import *
from clustering import *
from utils import *
from constant import PATH_OUTPUT, MODEL_CLUSTERING

def load_images_from_folder(folder):
    """
    Charge les images à partir d'un dossier et les retourne sous forme de tableau numpy.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Redimensionner l'image à 128x128 pixels
                images.append(img)
    return np.array(images)

def pipeline():
   
    images = load_images_from_folder("images")
    labels_true = np.zeros(len(images)) 
    print(images[0])

   
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features hog...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images)
    print("- calcul features HSV...")
    descriptors_hsv = compute_hsv_histograms(images)
    print("- calcul features SIFT...")
    descriptors_sift = compute_sift_descriptors(images)

    print("\n\n ##### Clustering ######")
    number_cluster = 10
    if MODEL_CLUSTERING == "kmeans":
        clustered_hog = KMEANS(number_cluster)
        clustered_hist = KMEANS(number_cluster)
        clustered_hsv = KMEANS(number_cluster)
        clustered_sift = KMEANS(number_cluster)
    elif MODEL_CLUSTERING == "dbscan":
        clustered_hog = DBSCAN()
        clustered_hist = DBSCAN()
        clustered_hsv = DBSCAN()
        clustered_sift = DBSCAN()
    else:
        raise ValueError("Modèle de clustering non supporté")

    print("- calcul "+MODEL_CLUSTERING+" avec features HOG ...")
    clustered_hog.fit(np.array(descriptors_hog))
    print("- calcul "+MODEL_CLUSTERING+" avec features Histogram...")
    clustered_hist.fit(np.array(descriptors_hist))
    print("- calcul "+MODEL_CLUSTERING+" avec features HSV...")
    clustered_hsv.fit(np.array(descriptors_hsv))
    print("- calcul "+MODEL_CLUSTERING+" avec features SIFT...")
    clustered_sift.fit(np.array(descriptors_sift))

    print("\n\n ##### Résultat ######")
    metric_hist = show_metric(labels_true, clustered_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", bool_return=True)
    metric_hog = show_metric(labels_true, clustered_hog.labels_, descriptors_hog,bool_show=True, name_descriptor="HOG", bool_return=True)
    metric_hsv = show_metric(labels_true, clustered_hsv.labels_, descriptors_hsv, bool_show=True, name_descriptor="HSV", bool_return=True)
    metric_sift = show_metric(labels_true, clustered_sift.labels_, descriptors_sift, bool_show=True, name_descriptor="SIFT", bool_return=True)

    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_hist,metric_hog,metric_hsv,metric_sift]
    df_metric = pd.DataFrame(list_dict)
    
    # Normalisation des données
    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)
    descriptors_hsv_norm = scaler.fit_transform(descriptors_hsv)
    descriptors_sift_norm = scaler.fit_transform(descriptors_sift)
    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)
    x_3d_hsv = conversion_3d(descriptors_hsv_norm)
    x_3d_sift = conversion_3d(descriptors_sift_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_hist = create_df_to_export(x_3d_hist, labels_true, clustered_hist.labels_)
    df_hog = create_df_to_export(x_3d_hog, labels_true, clustered_hog.labels_)
    df_hsv = create_df_to_export(x_3d_hsv, labels_true, clustered_hsv.labels_)
    df_sift = create_df_to_export(x_3d_sift, labels_true, clustered_sift.labels_)


    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT):
        # Crée le dossier
        os.makedirs(PATH_OUTPUT)

    # sauvegarde des données
    df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist.xlsx", index=False, engine='openpyxl')
    df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog.xlsx", index=False, engine='openpyxl')
    df_hsv.to_excel(PATH_OUTPUT + "/save_clustering_hsv.xlsx", index=False, engine='openpyxl')
    df_sift.to_excel(PATH_OUTPUT + "/save_clustering_sift.xlsx", index=False, engine='openpyxl')


    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx", index=False, engine='openpyxl')
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()