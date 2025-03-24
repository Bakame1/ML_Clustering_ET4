from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics
from constant import MODEL_CLUSTERING

# Algorithme KMeans existant
class KMEANS:
    def __init__(self, n_clusters, max_iter=300, random_state=None):
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
        """
        np.random.seed(self.random_state)  # Pour reproductibilité
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

    def nearest_cluster(self, X):
        """
        Calcule la distance euclidienne entre chaque point de X et les centres de clusters,
        et retourne l'indice du cluster le plus proche.
        """
        distances = []
        for centre in self.cluster_centers_:
            dist = np.sqrt(((X - centre)**2).sum(axis=1))
            distances.append(dist)
        distances = np.array(distances)
        return np.argmin(distances, axis=0)

    def fit(self, X):
        self.initialize_centers(X)
        for _ in range(self.max_iter):
            # Étape 1 : Assignation
            self.labels_ = self.nearest_cluster(X)
            # Étape 2 : Mise à jour des centres
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


class KMEDOIDS:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        """
        Initialise un objet KMedoids.

        Entrées:
        - n_clusters (int): Le nombre de clusters à former.
        - max_iter (int): Le nombre maximum d'itérations (par défaut 100).
        - random_state (int ou None): La graine pour la reproductibilité.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.medoids_ = None

    def initialize_medoids(self, X):
        """
        Initialise aléatoirement les médianes à partir de X.
        """
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def assign_clusters(self, X, medoids):
        """
        Assigne chaque point de X au médoïde le plus proche.
        """
        distances = np.array([[np.linalg.norm(point - medoid) for medoid in medoids] for point in X])
        return np.argmin(distances, axis=1)

    def update_medoids(self, X, clusters):
        """
        Met à jour les médianes pour chaque cluster en minimisant le coût intra-cluster.
        """
        new_medoids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[clusters == i]
            if len(cluster_points) == 0:
                # Si aucun point n'est assigné, on garde l'ancienne médiane
                continue
            # Calculer le coût total pour chaque point du cluster
            costs = np.array([np.sum(np.linalg.norm(cluster_points - point, axis=1)) for point in cluster_points])
            medoid = cluster_points[np.argmin(costs)]
            new_medoids[i] = medoid
        return new_medoids

    def fit(self, X):
        """
        Entraîne l'algorithme k-medoids sur les données X.
        """
        medoids = self.initialize_medoids(X)
        for _ in range(self.max_iter):
            clusters = self.assign_clusters(X, medoids)
            new_medoids = self.update_medoids(X, clusters)
            if np.allclose(medoids, new_medoids):
                break  # Convergence atteinte
            medoids = new_medoids
        self.labels_ = self.assign_clusters(X, medoids)
        self.medoids_ = medoids

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# Fonction pour afficher les métriques de clustering
def show_metric(labels_true, labels_pred, descriptors, bool_return=False, name_descriptor="", name_model="kmeans", bool_show=True):
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = metrics.jaccard_score(labels_true, labels_pred, average='macro')
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
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
