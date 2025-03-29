import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from PIL import Image
from constant import PATH_OUTPUT

# --- Fonctions utilitaires pour le chargement des fichiers Excel ---

@st.cache_data
def load_excel(file_name):
    file_path = os.path.join(PATH_OUTPUT, file_name)
    if not os.path.exists(file_path):
        st.error(f"Le fichier {file_name} n'existe pas. Veuillez lancer le pipeline pour générer les données.")
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier {file_name} : {e}")
        return pd.DataFrame()

# Chargement des mappings et métriques se fait en fonction de l'algo sélectionné
@st.cache_data
def load_mapping(descriptor, algo):
    mapping_file = os.path.join(PATH_OUTPUT, f"image_clusters_{descriptor.lower()}_{algo}.csv")
    if not os.path.exists(mapping_file):
        st.error(f"Le mapping pour {descriptor} avec l'algo {algo} n'existe pas.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(mapping_file)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du mapping {mapping_file} : {e}")
        return pd.DataFrame()

# Chargement des métriques globales (pour tous les algos)
@st.cache_data
def load_metric(algo):
    metric_file = os.path.join(PATH_OUTPUT, "save_metric_all.xlsx")
    if not os.path.exists(metric_file):
        st.error("Le fichier des métriques n'existe pas. Veuillez lancer le pipeline pour générer les données.")
        return pd.DataFrame()
    try:
        df = pd.read_excel(metric_file, engine='openpyxl')
        # Filtrer les métriques pour l'algo sélectionné
        df_algo = df[df["algo"] == algo]
        return df_algo
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier des métriques : {e}")
        return pd.DataFrame()

# --- Fonctions de visualisation des clusters ---

@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster')
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(
        x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
        mode='markers', marker=dict(color='red', size=10),
        name=f'Cluster {selected_cluster}'
    )
    return fig

@st.cache_data
def plot_metric(df_metric):
    fig = px.bar(
        df_metric,
        x='descriptor',
        y='ami',
        color='descriptor',
        title='Comparaison du score AMI entre descripteurs',
        labels={'ami': 'Adjusted Mutual Info Score', 'descriptor': 'Type de descripteur'}
    )
    return fig

# --- Construction de l'interface Dashboard ---

st.title("Dashboard Clustering des données")

# Création de trois onglets : Analyse par descripteur, Analyse globale, Images par Cluster
tab1, tab2, tab3 = st.tabs(["Analyse par descripteur", "Analyse globale", "Images par Cluster"])

# Sidebar pour la sélection de l'algorithme, du descripteur et du cluster
with st.sidebar:
    st.header("Options d'analyse")
    selected_algo = st.selectbox('Sélectionner l\'algorithme de clustering', ["kmeans", "kmedoids", "agglomerative"])
    selected_descriptor = st.selectbox('Sélectionner un descripteur', ["HIST", "HOG", "HSV", "SIFT", "CNN", "RESNET"])
    selected_cluster = st.selectbox('Sélectionner un Cluster', list(range(20)))
    st.session_state.selected_algo = selected_algo
    st.session_state.selected_descriptor = selected_descriptor
    st.session_state.selected_cluster = selected_cluster

# Onglet 1 : Analyse par descripteur
with tab1:
    st.header("Résultat de Clustering par descripteur")
    descriptor = st.session_state.selected_descriptor
    algo = st.session_state.selected_algo
    # Charger le fichier 3D correspondant pour le descripteur et l'algo
    file_3d = os.path.join(PATH_OUTPUT, f"save_clustering_{descriptor.lower()}_{algo}.xlsx")
    df = load_excel(os.path.basename(file_3d))
    st.subheader(f"Analyse du descripteur {descriptor} - Cluster {st.session_state.selected_cluster} ({algo.upper()})")
    if not df.empty:
        fig = colorize_cluster(df, st.session_state.selected_cluster)
        st.plotly_chart(fig)
    else:
        st.warning("Aucune donnée disponible pour ce descripteur.")

# Onglet 2 : Analyse globale
with tab2:
    st.header("Analyse Globale des Descripteurs")
    algo = st.session_state.selected_algo
    df_metric = load_metric(algo)
    if not df_metric.empty:
        fig_metric = plot_metric(df_metric)
        st.plotly_chart(fig_metric)
        st.subheader("Métriques")
        st.dataframe(
            df_metric[['descriptor', 'ami', 'ari', 'v_measure', 'silhouette']]
            .style
            .highlight_max(color='lightgreen', subset=['ami', 'ari', 'v_measure', 'silhouette'])
            .format("{:.2f}", subset=['ami', 'ari', 'v_measure', 'silhouette']),
            height=200
        )
    else:
        st.warning("Les métriques ne sont pas disponibles pour cet algorithme.")

# Onglet 3 : Images par Cluster
with tab3:
    st.header("Images par Cluster")
    descriptor = st.session_state.selected_descriptor
    algo = st.session_state.selected_algo
    cluster_id = st.session_state.selected_cluster
    st.subheader(f"Images du descripteur {descriptor} - Cluster {cluster_id} ({algo.upper()})")
    df_mapping = load_mapping(descriptor, algo)
    if not df_mapping.empty:
        df_cluster = df_mapping[df_mapping["predicted_cluster"] == cluster_id]
        nb_colonnes = 4
        colonnes = st.columns(nb_colonnes)
        for idx, row in df_cluster.iterrows():
            image_path = row["image_path"]
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    colonnes[idx % nb_colonnes].image(image, caption=os.path.basename(image_path), use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de l'ouverture de l'image {image_path} : {e}")
            else:
                st.warning(f"Le fichier {image_path} est introuvable.")
    else:
        st.info("Le mapping image-cluster n'est pas disponible.")
