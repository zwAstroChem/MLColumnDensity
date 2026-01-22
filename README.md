This repository contains two Python scripts for predicting molecular column densities (log10 scale) using machine learning regressors. The models are trained on previously detected molecules and then applied to candidate species. Two different molecular representations are supported:
Mol2Vec embeddings (70-dimensional continuous vectors)
ECFP fingerprints with charge encoding (binary fingerprints + charge one-hot features)
Both workflows follow the same strategy: repeated random subsampling of the training set, multi-model regression, and ensemble statistics (mean and standard deviation) over multiple runs.

---------------------------
Instructions for using the code
---------------------------

1. Predict_column_density_Mol2vec_ml.py

Purpose
Predict column densities using precomputed Mol2Vec embeddings.

Key characteristics

Input features: 70-dimensional Mol2Vec vectors

Target: log10(column density)

Models used:

Gradient Boosting Regressor (GBR)

Support Vector Regressor (SVR)

k-Nearest Neighbors Regressor (KNN)

Random Forest Regressor (RFR)

Training strategy: 10 independent runs, each using a random 70% subsample of the training set

Input files

columndensity_train_enhanced_Mol2Vec.txt
candidate_Mol2Vec.txt


Output For each model, a file is generated:
prediction_<MODEL>.txt

Each output file contains:
SMILES

log10 predictions from each of the 10 runs

Mean prediction across runs

Standard deviation across runs


2. Predict_column_density_ECFP_ml.py

Purpose
Predict column densities using ECFP fingerprints combined with molecular charge information.

Key characteristics

Input features: ECFP fingerprint (2048 bits)

Charge encoding: one-hot representation of charge (-1, 0, +1), replicated and concatenated

Final feature vector: [charge_bits + ECFP_bits]

Models and training strategy: identical to the Mol2Vec workflow

Input files

columndensity_train_enhanced_ECFP.txt
Format (space-separated):

SMILES  charge  log10_column_density  bit1 bit2 bit3 ...

candidate_ECFP.txt

Output For each model, a file is generated:

prediction_<MODEL>.txt

with the same structure as in the Mol2Vec case (10 runs + mean + standard deviation).


--------------------------
Dependencies
--------------------------

Python >= 3.8

NumPy

pandas

scikit-learntion:
pip install numpy pandas scikit-learn
