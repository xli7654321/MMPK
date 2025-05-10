import os
import sys
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

def split(df, col, output_dir, n_splits=10, test_size=1/9, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    unique_values = df[col].drop_duplicates()
    print(f"Number of unique {col}: {len(unique_values)}")

    for i, (train_idx, test_idx) in enumerate(kf.split(unique_values)):
        train_values, test_values = unique_values.iloc[train_idx], unique_values.iloc[test_idx]
        train_values, val_values = train_test_split(train_values, test_size=test_size, random_state=seed)
        print(f"Fold {i+1}: train_{col} {len(train_values)}, val_{col} {len(val_values)}, test_{col} {len(test_values)}")

        train_data = df[df[col].isin(train_values)]
        val_data = df[df[col].isin(val_values)]
        test_data = df[df[col].isin(test_values)]
        print(f"Fold {i+1}: train_data {len(train_data)}, val_data {len(val_data)}, test_data {len(test_data)}")

        fold_dir = os.path.join(output_dir, f"fold{i+1}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        train_data.to_csv(os.path.join(fold_dir, f"train_data_fold_{i+1}.csv"), index=False)
        val_data.to_csv(os.path.join(fold_dir, f"val_data_fold_{i+1}.csv"), index=False)
        test_data.to_csv(os.path.join(fold_dir, f"test_data_fold_{i+1}.csv"), index=False)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    df = pd.read_csv('data/approved/approved_modeling.csv')
    output_dir = 'data/approved'
    split(df, 'smiles', output_dir, n_splits=10, test_size=1/9, seed=42)
