from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from constant import  MODEL_CLUSTERING


class KMEANS:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Initialise un objet KMeans.

        Entrées:
        - n_clusters (int): Le nombre de clusters à former (par défaut 8).
        - max_iter (int): Le nombre maximum d'itérations pour l'algorithme (par défaut 300).
        - random_state (int ou None): La graine pour initialiser le générateur de nombres aléatoires (par défaut None).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize_centers(self, X):
        """
        Initialise les centres de clusters avec n_clusters points choisis aléatoirement à partir des données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """

        np.random.seed(self.random_state)#Pour avoir des centres differents celon les executions
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

        pass

    def nearest_cluster(self, X):
        """
        Calcule la distance euclidienne entre chaque point de X et les centres de clusters,
        puis retourne l'indice du cluster le plus proche pour chaque point.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster le plus proche pour chaque point.
        """
        
        distances = []
        for centre in self.cluster_centers_:
            dist = np.sqrt(((X - centre)**2).sum(axis=1))  # Distance euclidienne par point
            distances.append(dist)
        
        distances = np.array(distances)
        return np.argmin(distances, axis=0)  #axis=0 pour un tableau


    def fit(self, X):
        self.initialize_centers(X)
        
        for _ in range(self.max_iter):
            # Étape 1 : Assignation aux clusters
            self.labels_ = self.nearest_cluster(X)
            
            # Étape 2 : Mise à jour des centres
            nouveaux_centres = []
            for cluster_i in range(self.n_clusters):
                points_du_cluster = X[self.labels_ == cluster_i]
                if len(points_du_cluster) > 0:
                    nouveaux_centres.append(points_du_cluster.mean(axis=0))
                else:
                    nouveaux_centres.append(self.cluster_centers_[cluster_i])
            
            # Vérification de la convergence
            if np.allclose(nouveaux_centres, self.cluster_centers_):
                break
                
            self.cluster_centers_ = np.array(nouveaux_centres)
        
        # Mise à jour finale des labels avec les derniers centres
        self.labels_ = self.nearest_cluster(X)
                

        def predict(self, X):
            """
            Prédit l'appartenance aux clusters pour les données X en utilisant les centres de clusters appris pendant l'entraînement.

            Entrée:
            - X (np.array): Les données d'entrée.

            Sortie:
            - np.array: Un tableau d'indices représentant le cluster prédit pour chaque point.
            """
            return self.nearest_cluster(X)



class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialise un objet DBSCAN.

        Entrées:
        - eps (float): La distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins.
        - min_samples (int): Le nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un point central.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = dbscan.fit_predict(X)

    def fit_predict(self, X):
        return self.labels_

    
def show_metric(labels_true, labels_pred, descriptors,bool_return=False,name_descriptor="", name_model="kmeans",bool_show=True):
    """
    Fonction d'affichage et création des métrique pour le clustering.
    Input :
    - labels_true : étiquettes réelles des données
    - labels_pred : étiquettes prédites des données
    - descriptors : ensemble de descripteurs utilisé pour le clustering
    - bool_return : booléen indiquant si les métriques doivent être retournées ou affichées
    - name_descriptor : nom de l'ensemble de descripteurs utilisé pour le clustering
    - name_model : nom du modèle de clustering utilisé
    - bool_show : booléen indiquant si les métriques doivent être affichées ou non

    Output :
    - dictionnaire contenant les métriques d'évaluation des clusters
    """
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = metrics.jaccard_score(labels_true, labels_pred, average='macro')
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    silhouette = silhouette_score(descriptors, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    # Affichons les résultats
    if bool_show :
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Mutual Information: {ami}")
    if bool_return:
        return {"ami":ami,
                "ari":ari, 
                "silhouette":silhouette,
                "homogeneity":homogeneity,
                "completeness":completeness,
                "v_measure":v_measure, 
                "jaccard":jaccard,
               "descriptor":name_descriptor,
               "name_model":name_model}
    