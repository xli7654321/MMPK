import pandas as pd
from pathlib import Path
import argparse

def main(folder_path):
    folder_path = Path(folder_path)

    csv_files = list(folder_path.glob('*.csv'))

    dfs = [pd.read_csv(file) for file in csv_files]
    all_folds_df = pd.concat(dfs, axis=0, ignore_index=True)
    all_folds_df = all_folds_df.sort_values(by=['Fold'])
    
    mean_results = all_folds_df.groupby(['Task']).mean().reset_index()
    mean_results = mean_results.drop(columns=['Epoch', 'Fold'])

    std_results = all_folds_df.groupby(['Task']).std(ddof=1).reset_index()
    std_results = std_results.drop(columns=['Epoch', 'Fold'])

    task_order = ['auc', 'cmax', 'tmax', 'hl', 'cl', 'vz', 'mrt', 'f']
    
    for df in [mean_results, std_results]:
        df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)
        df.sort_values(by=['Task'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    first_file = csv_files[0].name
    prefix = first_file.rsplit('_', 1)[0]
    
    mean_output_file = folder_path / f"{prefix}_mean.csv"
    std_output_file = folder_path / f"{prefix}_std.csv"

    mean_results.to_csv(mean_output_file, index=False)
    std_results.to_csv(std_output_file, index=False)

    print(f"Saved mean results to: {mean_output_file}")
    print(f"Saved std results to: {std_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, required=True)
    args = parser.parse_args()
    main(args.folder_path)
