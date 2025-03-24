import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from constant import PATH_OUTPUT, MODEL_CLUSTERING

# Fonction utilitaire pour charger un fichier Excel en gérant les erreurs
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

# Chargement des fichiers Excel
df_hist   = load_excel("save_clustering_hist.xlsx")
df_hog    = load_excel("save_clustering_hog.xlsx")
df_hsv    = load_excel("save_clustering_hsv.xlsx")
df_sift   = load_excel("save_clustering_sift.xlsx")
df_metric = load_excel("save_metric.xlsx")

if not df_metric.empty and 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Fonction pour créer un graphique 3D coloré en fonction du cluster sélectionné
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

# Fonction pour afficher un histogramme comparatif des scores AMI
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

st.title("Dashboard Clustering des données")

# Création de deux onglets pour l'analyse
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse globale"])

with tab1:
    st.header("Résultat de Clustering par descripteur")
    st.sidebar.header("Options d'analyse")
    # Sélection du descripteur à analyser
    descriptor = st.sidebar.selectbox('Sélectionner un descripteur', ["HIST", "HOG", "HSV","SIFT"])
    if descriptor == "HIST":
        df = df_hist
    elif descriptor == "HOG":
        df = df_hog
    elif descriptor == "HSV":
        df = df_hsv
    elif descriptor == "SIFT":
        df = df_sift

    # Sélection du cluster à analyser
    selected_cluster = st.sidebar.selectbox('Sélectionner un Cluster', range(10))
    st.subheader(f"Analyse du descripteur {descriptor} - Cluster {selected_cluster}")

    if not df.empty:
        fig = colorize_cluster(df, selected_cluster)
        st.plotly_chart(fig)
    else:
        st.warning("Aucune donnée disponible pour ce descripteur.")

with tab2:
    st.header("Analyse Globale des Descripteurs")
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
        st.warning("Les métriques ne sont pas disponibles.")
