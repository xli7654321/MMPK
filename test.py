import os
import torch
import pandas as pd
from config import args
from dataloader import DataLoaderSelector
from mmpk import MMPKPredictor
from utils import *

TASKS = [param for param in args.pk_params]

def test(model, loader, device):
    model.eval()

    preds = {task: [] for task in TASKS}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, label, dose = [item.to(device) for item in batch]

            # scores.shape: (batch_size, num_subs)
            y_hat, _ = model(mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, dose)
            y = label.view(y_hat.shape)  # (batch_size, num_tasks)
            
            # Store predictions for this fold by task
            for mol_idx, dose_item, y_hat_item, y_item in zip(range(len(mol_gdata)),
                                                              dose.detach().cpu(),
                                                              y_hat.detach().cpu(),
                                                              y.detach().cpu()):
                for task_idx, task in enumerate(TASKS):
                    preds[task].append({
                        'smiles': mol_gdata[mol_idx].smi,
                        'dose': list(dose_item.unsqueeze(0).numpy())[0],
                        'y_hat': list(y_hat_item[task_idx].unsqueeze(0).numpy())[0],
                        'y': list(y_item[task_idx].unsqueeze(0).numpy())[0]
                    })

    return preds

if __name__ == '__main__':
    seed_everything(args.seed)
    device = torch.device(
        'cuda:' + str(args.device)
        if torch.cuda.is_available() else 'cpu'
    )
    num_tasks = len(args.pk_params)

    ext_invest_preds_by_task = {task: [] for task in TASKS}
    ext_2024_preds_by_task = {task: [] for task in TASKS}
    
    for fold in range(1, 11):
        model = MMPKPredictor(args, num_tasks=num_tasks)
        model.to(device)
    
        model_path = f"checkpoints/{args.checkpoints_folder}/fold_{fold}.pth"
        print(f"Loading MMPK model {fold} from checkpoints/{args.checkpoints_folder} for evaluation on external validation set...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        selector = DataLoaderSelector(fold, args.pk_params, load_ext=True)
        ext_invest_loader, ext_2024_loader = selector.select_loader(loader_type='ext')
        
        ext_invest_preds = test(model, ext_invest_loader, device)
        ext_2024_preds = test(model, ext_2024_loader, device)
    
        for task in TASKS:
            df_invest = pd.DataFrame(ext_invest_preds[task])
            df_2024 = pd.DataFrame(ext_2024_preds[task])
            ext_invest_preds_by_task[task].append(df_invest)
            ext_2024_preds_by_task[task].append(df_2024)
    
    output_dir_invest = 'results/investigational'
    output_dir_2024 = 'results/2024'
    os.makedirs(output_dir_invest, exist_ok=True)
    os.makedirs(output_dir_2024, exist_ok=True)
    
    metrics_invest = []
    metrics_2024 = []
    
    for task in TASKS:
        print(f"Processing {task} for investigational set...")
        dfs_invest = ext_invest_preds_by_task[task]
        
        merged_df_invest = pd.concat([
            df_invest[['y_hat']].rename(columns={'y_hat': f"y_hat_fold_{i+1}"})
            for i, df_invest in enumerate(dfs_invest)
        ], axis=1)
        merged_df_invest['y_hat_mean'] = merged_df_invest.mean(axis=1)
        
        ref_df_invest = dfs_invest[0][['y', 'dose', 'smiles']].reset_index(drop=True)
        merged_df_invest = pd.concat([merged_df_invest, ref_df_invest], axis=1)
        
        cols = [f"y_hat_fold_{i+1}" for i in range(10)] + ['y_hat_mean', 'y', 'dose', 'smiles']
        merged_df_invest = merged_df_invest[cols]
        
        # Save predictions
        merged_df_invest.to_csv(os.path.join(output_dir_invest, f"ext_invest_preds_{task}.csv"), index=False)
        
        # Calculate metrics
        merged_df_invest = merged_df_invest.dropna(subset=['y'])
        y_invest, y_hat_invest = merged_df_invest['y'], merged_df_invest['y_hat_mean']
        
        gmfe_invest = gmfe(y_invest, y_hat_invest, pk_param=task)
        rmsle_invest = rmsle(y_invest, y_hat_invest)
        afe_invest = afe(y_invest, y_hat_invest, pk_param=task)
        pearson_r_invest = pearson_r(y_invest, y_hat_invest)
        
        print("Investigational set:")
        print(f"PK parameter: {task}")
        print(f"GMFE: {gmfe_invest}")
        print(f"RMSE: {rmsle_invest}")
        print(f"AFE: {afe_invest}")
        print(f"Pearson R: {pearson_r_invest}")
        
        metrics_invest.append({
            'task': task,
            'gmfe': gmfe_invest,
            'rmsle': rmsle_invest,
            'afe': afe_invest,
            'pearson_r': pearson_r_invest
        })
        
        print(f"Processing {task} for 2024 FDA-approved set...")
        dfs_2024 = ext_2024_preds_by_task[task]
        
        merged_df_2024 = pd.concat([
            df_2024[['y_hat']].rename(columns={'y_hat': f"y_hat_fold_{i+1}"})
            for i, df_2024 in enumerate(dfs_2024)
        ], axis=1)
        merged_df_2024['y_hat_mean'] = merged_df_2024.mean(axis=1)
        
        ref_df_2024 = dfs_2024[0][['y', 'dose', 'smiles']].reset_index(drop=True)
        merged_df_2024 = pd.concat([merged_df_2024, ref_df_2024], axis=1)
        
        cols = [f"y_hat_fold_{i+1}" for i in range(10)] + ['y_hat_mean', 'y', 'dose', 'smiles']
        merged_df_2024 = merged_df_2024[cols]
        
        # Save predictions
        merged_df_2024.to_csv(os.path.join(output_dir_2024, f"ext_2024_preds_{task}.csv"), index=False)
        
        # Calculate metrics
        merged_df_2024 = merged_df_2024.dropna(subset=['y'])
        y_2024, y_hat_2024 = merged_df_2024['y'], merged_df_2024['y_hat_mean']
        
        gmfe_2024 = gmfe(y_2024, y_hat_2024, pk_param=task)
        rmsle_2024 = rmsle(y_2024, y_hat_2024)
        afe_2024 = afe(y_2024, y_hat_2024, pk_param=task)
        pearson_r_2024 = pearson_r(y_2024, y_hat_2024)
        
        print("2024 FDA-approved set:")
        print(f"PK parameter: {task}")
        print(f"GMFE: {gmfe_2024}")
        print(f"RMSE: {rmsle_2024}")
        print(f"AFE: {afe_2024}")
        print(f"Pearson R: {pearson_r_2024}")
        
        metrics_2024.append({
            'task': task,
            'gmfe': gmfe_2024,
            'rmsle': rmsle_2024,
            'afe': afe_2024,
            'pearson_r': pearson_r_2024
        })
    
    # Save metrics
    pd.DataFrame(metrics_invest).to_csv(os.path.join(output_dir_invest, f"ext_invest_metrics.csv"), index=False)
    pd.DataFrame(metrics_2024).to_csv(os.path.join(output_dir_2024, f"ext_2024_metrics.csv"), index=False)
    