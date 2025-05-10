import os
import pandas as pd

def save_results(path, n_fold, pk_params, test_metrics, epoch):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    rows = []
    for task_idx, task_name in enumerate(pk_params):
        row_test = {
            'Epoch': epoch,
            'Fold': n_fold,
            'Task': task_name
        }
        for key, value in test_metrics.items():
            row_test[key] = value[task_idx]
        rows.append(row_test)

    df = pd.DataFrame(rows)
    df.to_csv(path, mode='w', header=True, index=False)

if __name__ == '__main__':
    pass
