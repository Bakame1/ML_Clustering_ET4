from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import jaccard_score, adjusted_rand_score

# ------------------------------------------------------------------
# Algorithme KMeans
# ------------------------------------------------------------------
class KMEANS:
    def __init__(self, n_clusters, max_iter=300, random_state=None):
        #Initialisation des paramètres : nombre de clusters, itérations max et seed de randomisation
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None  #Centres des clusters (mis à jour pendant l'entraînement)
        self.labels_ = None           #Affectation de chaque point à un cluster

    def initialize_centers(self, X):
        #On fixe la graine pour la reproductibilité, puis on choisit aléatoirement n_clusters points dans X
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

    def nearest_cluster(self, X):
        #Pour chaque centre, on calcule la distance euclidienne entre les points de X et du centre
        distances = []
        for centre in self.cluster_centers_:
            #distance euclidienne
            dist = np.sqrt(((X - centre)**2).sum(axis=1))
            distances.append(dist)
        distances = np.array(distances)

        #Retourne l'indice du cluster le plus proche pour chaque point
        return np.argmin(distances, axis=0)

    def fit(self, X):
        #Initialisation des centres
        self.initialize_centers(X)

        for _ in range(self.max_iter):
            #Affectation des points aux clusters
            self.labels_ = self.nearest_cluster(X)
            nouveaux_centres = []
            #Mise à jour des centres par calcul de la moyenne de chaque cluster
            for cluster_i in range(self.n_clusters):
                points_du_cluster = X[self.labels_ == cluster_i]
                if len(points_du_cluster) > 0:
                    nouveaux_centres.append(points_du_cluster.mean(axis=0))
                else:
                    #Si un cluster n'a aucun point, on conserve son ancien centre
                    nouveaux_centres.append(self.cluster_centers_[cluster_i])
            #si les nouveaux centres sont très proches des anciens (convergence)
            if np.allclose(nouveaux_centres, self.cluster_centers_):
                break
            self.cluster_centers_ = np.array(nouveaux_centres)
        #Affectation finale des points aux clusters avec les centres optimisés
        self.labels_ = self.nearest_cluster(X)

    def predict(self, X):
        #Prédiction : on attribue chaque point au cluster le plus proche des centres appris
        return self.nearest_cluster(X)

# ------------------------------------------------------------------
# Algorithme KMedoids
# ------------------------------------------------------------------
class KMEDOIDS:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        #Initialisation du nombre de clusters, du nombre d'itérations et de la seed
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None     #Affectation des points aux clusters
        self.medoids_ = None    #Indices ou valeurs représentant les médianes de chaque cluster

    def initialize_medoids(self, X):
        #Choix aléatoire des points initiaux comme médianes
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def assign_clusters(self, X, medoids):
        #Pour chaque point, calcule la distance par rapport à chaque médiane et affecte le cluster le plus proche
        distances = np.array([[np.linalg.norm(point - medoid) for medoid in medoids] for point in X])
        return np.argmin(distances, axis=1)

    def update_medoids(self, X, clusters):
        #Met à jour les médianes de chaque cluster en cherchant le point du cluster qui minimise la somme des distances
        new_medoids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[clusters == i]
            if len(cluster_points) == 0:
                #Si aucun point n'est assigné à ce cluster, on ne change pas la médiane
                continue
            #Calcul du coût total pour chaque point dans le cluster
            costs = np.array([np.sum(np.linalg.norm(cluster_points - point, axis=1)) for point in cluster_points])
            medoid = cluster_points[np.argmin(costs)]
            new_medoids[i] = medoid
        return new_medoids

    def fit(self, X):
        #Initialisation des médianes(medoids)
        medoids = self.initialize_medoids(X)

        for _ in range(self.max_iter):
            clusters = self.assign_clusters(X, medoids)
            new_medoids = self.update_medoids(X, clusters)
            if np.allclose(medoids, new_medoids):
                #Convergence atteinte si les médianes ne changent plus
                break
            medoids = new_medoids
        #Affectation finale des points
        self.labels_ = self.assign_clusters(X, medoids)
        self.medoids_ = medoids

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# ------------------------------------------------------------------
# Algorithme Agglomératif (Hierarchical Clustering)
# ------------------------------------------------------------------
class Agglomerative_Clustering:
    def __init__(self, n_clusters, linkage='ward'):
        #n_clusters (int) : Nombre de clusters à trouver.
        #linkage (str) : Méthode de linkage 
        #       (par défaut 'ward'->minimise la somme des carrés des différences à l'intérieur des clusters)

        
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels_ = None

    def fit(self, X):
        #Applique le clustering hiérarchique et stocke les labels
        self.labels_ = self.model.fit_predict(X)

    def predict(self, X):
        #Pour l'agglomératif, on ne prédit pas sur de nouvelles données ; on renvoie simplement les labels calculés
        return self.labels_

# ------------------------------------------------------------------
# Fonction pour afficher les métriques de clustering
# ------------------------------------------------------------------
def show_metric(labels_true, labels_pred, descriptors, bool_return=False, name_descriptor="", name_model="", bool_show=True):
    #Calcul des métriques : homogeneity, completeness, v_measure, jaccard, ami, silhouette et ARI
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = jaccard_score(labels_true, labels_pred, average='macro')
    ami = normalized_mutual_info_score(labels_true, labels_pred)
    silhouette = silhouette_score(descriptors, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    if bool_show:
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Mutual Information: {ami}")
    if bool_return:
        return {"ami": ami,
                "ari": ari,
                "silhouette": silhouette,
                "homogeneity": homogeneity,
                "completeness": completeness,
                "v_measure": v_measure,
                "jaccard": jaccard,
                "descriptor": name_descriptor,
                "name_model": name_model}
