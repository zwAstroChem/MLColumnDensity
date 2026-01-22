import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ---------------- 读取训练集 ----------------
def load_training_data(file_path: str):
    smiles_list, X, y = [], [], []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 4:
                print(f"⚠️ Row {i} has insufficient columns; skipping.")
                continue

            smiles = parts[0]
            try:
                log10_cd = float(parts[2])
                vector = np.array([float(x) for x in parts[3:]], dtype=float)
            except ValueError:
                print(f"⚠️ Failed to convert numerical values for SMILES {smiles} in row {i}; skipping.")
                continue

            if len(vector) != 70:
                print(f"⚠️ Row {i} SMILES '{smiles}' has vector length {len(vector)}, not 70; skipping.")
                continue

            smiles_list.append(smiles)
            X.append(vector)
            y.append(log10_cd)

    return np.array(X), np.array(y), smiles_list

# ---------------- 读取候选集 ----------------
def load_candidate_data(file_path: str):
    smiles_list, X = [], []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 4:
                print(f"⚠️ Row {i} has insufficient columns; skipping.")
                continue

            smiles = parts[0]
            vector_str = parts[-1] 
            try:
                vector = np.array([float(x) for x in vector_str.split(",")], dtype=float)
            except ValueError:
                print(f"⚠️ Failed to parse vector for SMILES '{smiles}' in row {i}.")
                continue

            if len(vector) != 70:
                print(f"⚠️ Row {i} SMILES '{smiles}' has vector length {len(vector)}, expected 70; skipping.")
                continue

            smiles_list.append(smiles)
            X.append(vector)

    return np.array(X), smiles_list

# ---------------- 主函数 ----------------
def main():
    # 读取训练数据和候选数据
    X_all, y_all, train_smiles = load_training_data("columndensity_train_enhanced_Mol2Vec.txt")
    X_test, test_smiles = load_candidate_data("candidate_Mol2Vec.txt")

    print(f"Training set: {X_all.shape}, Test set: {X_test.shape}")

    # 定义多个模型
    model_builders = {
        "GBR": lambda: GradientBoostingRegressor(),
        "SVR": lambda: SVR(),
        "KNN": lambda: KNeighborsRegressor(),
        "RFR": lambda: RandomForestRegressor()
    }

    for name, builder in model_builders.items():
        print(f"Run the model: {name}")
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
        out_file = f"prediction_{name}_MOL2VEC.txt"
        out_df.to_csv(out_file, sep="\t", index=False, float_format="%.6f")
        print(f"✅ Prediction results for model {name} have been saved to {out_file}.")

if __name__ == "__main__":
    main()
