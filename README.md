## Projet de clustering des immages SNACK pour ET4 info Polytech Paris Saclay
#### Auteur : Marko Babic + Base du code venant du cours de ML de Polytech Paris Saclay

### Step 1 : Version de Python
    - Cas 1 : Avoir une version de python entre 3.8 et 3.11
        => installer les requierements : "pip install -r requierements.txt"
    - Cas 2 : Utiliser l'envirronnement 3.11 depuis le folder .venv dans le code
        => Pour ce faire dans un terminal depuis le repertoire du projet :
            .venv\Scripts\activate
            Puis s'assurer que les commandes qui suivent soit effectuées vià l'environnement
            Exemple du terminal une fois l'env utilisé : 
            (.venv) PS C:\Users...
            => Utiliser pip si les librairies sont mal installees : 
                "pip install -r requierements.txt"

### step 2 :  run de la pipeline clustering
    - se placer dans le dossier src dans le terminal
    - exécutez la commande : "python pipeline.py"
    
### step 3 : lancement du dashboard
    - se placer dans le dossier src dans le terminal
    - exécutez la commande : "streamlit run dashboard_clustering.py"
    - si il y a un probleme lors du lancement du dashboard avec les fichiers excels
        => supprimer tous les fichiers dans le dossier src/donnees puis refaire l'etape 2 et 3

### Annexe : Concernant les images
    - a. Lien des photos Snack : https://huggingface.co/datasets/Matthijs/snacks/tree/main
        => en fonction de la puissance de votre PC, je vous conseille d'utiliser seulement les données dans le dossier validation.
        => dans le dossier src/constant.py, s'assurer que la variable "PATH_DATA" est le chemin vers le dossier contenant les images à clusteriser.
