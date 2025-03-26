## Projet de clustering des immages SNACK pour ET4 info Polytech Paris Saclay
## Auteur : Marko Babic + Base du code venant du cours de ML de Polytech Paris Saclay

### Préréquis : Version de Python
    - Avoir une version de python entre 3.8 et 3.11
    - Ou alors utiliser l'envirronnement 3.11 depuis le folder .venv
        => Pour ce faire dans un terminal depuis le repertoire du projet :
            .venv\Scripts\activate
            Puis s'assurer que les commandes qui suivent soit effectuées vià l'environnement
            Exemple du terminal une fois l'env utilisé : 
            (.venv) PS C:\Users...

### step 1 : téléchargement des données et installation des packages
    - a. télécharger les données Snack : https://huggingface.co/datasets/Matthijs/snacks/tree/main
        => en fonction de la puissance de votre PC, je vous conseille d'utiliser seulement les données dans le dossier validation.
    - b. installer les requierements : "pip install -r requierements.txt"
### step 2 : configuration du chemin vers les donnés
    - dans le dossier src/constant.py, modifier la variable "PATH_DATA" par le chemin vers le dossier contenant les données à clusteriser.

### step 3 :  run de la pipeline clustering
    - se placer dans le dossier src dans le terminal
    - exécutez la commande : "python pipeline.py"
    
### step 4 : lancement du dashboard
    - se placer dans le dossier src dans le terminal
    - exécutez la commande : "streamlit run dashboard_clustering.py"