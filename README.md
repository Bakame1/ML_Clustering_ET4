## SNACK Image Clustering Project for ET4 Info Polytech Paris Saclay
#### Author: Marko Babic + Code base from the ML course at Polytech Paris Saclay
### Step 1: Python Version
- Case 1: Have a Python version between 3.8 and 3.11
      => install the requirements:
```
pip install -r requierements.txt
```
- Case 2: Use the 3.11 environment from the .venv folder in the code
  To do this, in a terminal from the project directory:
```
.venv\Scripts\activate
```
Then make sure the following commands are executed via the environment
Example of the terminal once the env is used:
(.venv) PS C:\Users...
    => Use pip if the libraries are not installed correctly:
```
pip install -r requierements.txt
```
### Step 2: Run the Clustering Pipeline
- Place yourself in the src folder in the terminal
- Execute the command:

### Step 3: Launch the Dashboard
- Place yourself in the src folder in the terminal
- Execute the command:
- If there is a problem when launching the dashboard with the Excel files
      => delete all files in the src/donnees folder then redo steps 2 and 3
  
### Appendix: Regarding the Images
- Link to Snack photos: https://huggingface.co/datasets/Matthijs/snacks/tree/main
        => depending on the power of your PC, I advise you to use only the data in the validation folder.
        => in the src/constant.py folder, make sure that the "PATH_DATA" variable is the path to the folder containing the images to be clustered.
