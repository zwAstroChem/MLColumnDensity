import numpy as np 
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Convert charge to one-hot encoding (replicated n times → 3n-dimensional)
def charge_to_bits(charge: int) -> np.ndarray:
    if charge == -1:
        base = [1, 0, 0]
    elif charge == 0:
        base = [0, 1, 0]
    elif charge == 1:
        base = [0, 0, 1]
    else:
        base = [0, 0, 0]
    return np.tile(base, 5)

# Convert active bits to an ECFP fingerprint vector (2048 bits)
def activebits_to_ecfp(active_bits, n_bits=2048):
    arr = np.zeros((n_bits,), dtype=int)
    for b in active_bits:
        if 0 <= b < n_bits:
            arr[b] = 1
    return arr


def load_training_data(file_path: str):
    smiles_list, X, y = [], [], []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            smiles = parts[0]
            charge = int(parts[1])
            log10_cd = float(parts[2])
            active_bits = [int(x) for x in parts[3:]]

            charge_bits = charge_to_bits(charge)
            ecfp = activebits_to_ecfp(active_bits)
            features = np.concatenate([charge_bits, ecfp])

            smiles_list.append(smiles)
            X.append(features)
            y.append(log10_cd)
    return np.array(X), np.array(y), smiles_list


def load_candidate_data(file_path: str):
    smiles_list, X = [], []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            smiles = parts[0]
            charge = int(parts[1])
            active_bits = [int(x) for x in parts[2:]]

            charge_bits = charge_to_bits(charge)
            ecfp = activebits_to_ecfp(active_bits)
            features = np.concatenate([charge_bits, ecfp])

            smiles_list.append(smiles)
            X.append(features)
    return np.array(X), smiles_list

def main():
    
    X_all, y_all, train_smiles = load_training_data("columndensity_train_enhanced_ECFP.txt")
    X_test, test_smiles = load_candidate_data("candidate_ECFP.txt")

    print(f"Training set: {X_all.shape}, Test set: {X_test.shape}")

    
    model_builders = {
        "GBR": lambda: GradientBoostingRegressor(),
        "SVR": lambda: SVR(),
        "KNN": lambda: KNeighborsRegressor(),
        "RFR": lambda: RandomForestRegressor()
    }

    for name, builder in model_builders.items():
        print(f"Running model: {name}")
        results = {"SMILES": test_smiles}
        all_preds = []

        
        for run in range(10):
            
            n_samples = int(0.7 * len(X_all))
            idx = np.random.choice(len(X_all), size=n_samples, replace=False)
            X_train, y_train = X_all[idx], y_all[idx]

            model = builder()
            try:
                model.fit(X_train, y_train)
                y_pred_log10 = model.predict(X_test)
                y_pred_log10 = np.round(y_pred_log10, 6)
                results[f"log10_pred_run{run+1}"] = y_pred_log10
                all_preds.append(y_pred_log10)
            except Exception as e:
                print(f"⚠️ Model {name} failed on run {run + 1}: {e}")
                results[f"log10_pred_run{run+1}"] = [np.nan] * len(test_smiles)
                all_preds.append([np.nan] * len(test_smiles))

        
        all_preds = np.array(all_preds)
        mean_pred = np.round(np.nanmean(all_preds, axis=0), 6)
        std_pred = np.round(np.nanstd(all_preds, axis=0), 6)

        results["mean_pred"] = mean_pred
        results["std_pred"] = std_pred

        
        out_df = pd.DataFrame(results)
        out_file = f"prediction_{name}.txt"
        out_df.to_csv(out_file, sep="\t", index=False, float_format="%.6f")
        print(f"✅ Prediction results for model {name} have been saved to {out_file}.")

if __name__ == "__main__":
    main()
