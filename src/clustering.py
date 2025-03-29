from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics
from constant import MODEL_CLUSTERING
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import jaccard_score, adjusted_rand_score

# Algorithme KMeans existant
class KMEANS:
    def __init__(self, n_clusters, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize_centers(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

    def nearest_cluster(self, X):
        distances = []
        for centre in self.cluster_centers_:
            dist = np.sqrt(((X - centre)**2).sum(axis=1))
            distances.append(dist)
        distances = np.array(distances)
        return np.argmin(distances, axis=0)

    def fit(self, X):
        self.initialize_centers(X)
        for _ in range(self.max_iter):
            self.labels_ = self.nearest_cluster(X)
            nouveaux_centres = []
            for cluster_i in range(self.n_clusters):
                points_du_cluster = X[self.labels_ == cluster_i]
                if len(points_du_cluster) > 0:
                    nouveaux_centres.append(points_du_cluster.mean(axis=0))
                else:
                    nouveaux_centres.append(self.cluster_centers_[cluster_i])
            if np.allclose(nouveaux_centres, self.cluster_centers_):
                break
            self.cluster_centers_ = np.array(nouveaux_centres)
        self.labels_ = self.nearest_cluster(X)

    def predict(self, X):
        return self.nearest_cluster(X)

# Algorithme KMedoids existant
class KMEDOIDS:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.medoids_ = None

    def initialize_medoids(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def assign_clusters(self, X, medoids):
        distances = np.array([[np.linalg.norm(point - medoid) for medoid in medoids] for point in X])
        return np.argmin(distances, axis=1)

    def update_medoids(self, X, clusters):
        new_medoids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[clusters == i]
            if len(cluster_points) == 0:
                continue
            costs = np.array([np.sum(np.linalg.norm(cluster_points - point, axis=1)) for point in cluster_points])
            medoid = cluster_points[np.argmin(costs)]
            new_medoids[i] = medoid
        return new_medoids

    def fit(self, X):
        medoids = self.initialize_medoids(X)
        for _ in range(self.max_iter):
            clusters = self.assign_clusters(X, medoids)
            new_medoids = self.update_medoids(X, clusters)
            if np.allclose(medoids, new_medoids):
                break
            medoids = new_medoids
        self.labels_ = self.assign_clusters(X, medoids)
        self.medoids_ = medoids

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# Algorithme Agglomératif (Hierarchical Clustering)
class Agglomerative_Clustering:
    def __init__(self, n_clusters, linkage='ward'):
        """
        Initialise un objet Agglomerative Clustering.
        
        Entrées:
        - n_clusters (int) : Nombre de clusters à trouver.
        - linkage (str) : Méthode de linkage (par défaut 'ward').
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels_ = None

    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)

    def predict(self, X):
        # Pour l'agglomératif, la prédiction se fait sur le même jeu de données qu'on a fit.
        return self.labels_

# Fonction pour afficher les métriques de clustering
def show_metric(labels_true, labels_pred, descriptors, bool_return=False, name_descriptor="", name_model=MODEL_CLUSTERING, bool_show=True):
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
