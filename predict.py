import os
import time
from pathlib import Path
import torch
import pandas as pd
from config import args
from dataloader import MMPKPredictLoader
from mmpk import MMPKPredictor
from utils import *

TASKS = [param for param in args.pk_params]

def test(model, loader, device):
    model.eval()

    preds = {task: [] for task in TASKS}
    att_info = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, dose, orig_dose = [item.to(device) for item in batch]

            # scores.shape: (batch_size, num_subs)
            y_hat, scores = model(mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, dose)
            
            # Store predictions for this fold by task
            for mol_idx, dose_item, y_hat_item in zip(range(len(mol_gdata)),
                                                      orig_dose.detach().cpu(),
                                                      y_hat.detach().cpu()):
                for task_idx, task in enumerate(TASKS):
                    preds[task].append({
                        'smiles': mol_gdata[mol_idx].smi,
                        'dose': list(dose_item.unsqueeze(0).numpy())[0],
                        'y_hat': list(y_hat_item[task_idx].unsqueeze(0).numpy())[0]
                    })
            
            # substructure-level attention scores
            for mol_idx in range(scores.size(0)):
                sub_indices = (sub_mask[mol_idx] == 1).nonzero(as_tuple=True)[0].tolist()
                
                sub_scores = []
                for sub_idx in sub_indices:
                    sub_scores.append({
                        'sub_index': sub_idx,
                        'sub_smiles': sub_gdata[sub_idx].smi,
                        'sub_score': list(scores[mol_idx, sub_idx].detach().cpu().unsqueeze(0).numpy())[0]
                    })
            
                att_info.append({
                    'smiles': mol_gdata[mol_idx].smi,
                    'dose': list(orig_dose[mol_idx].detach().cpu().unsqueeze(0).numpy())[0],
                    'sub_scores': sub_scores
                })

    return preds, att_info

def get_user_input():
    print('\033[105mPlease select input mode [1: Manually, 2: CSV file] (default: 1):\033[0m')
    input_mode = input('> ').strip()
    if input_mode in ['', '1']:
        print('\033[93mPlease enter SMILES (separated by commas):\033[0m')
        smi_input = input('> ').strip()
        smi_list = [s.strip() for s in smi_input.split(',') if s.strip()]

        print('\033[93mPlease enter administered dose (separated by commas):\033[0m')
        dose_input = input('> ').strip()
        doses = [float(d) for d in dose_input.split(',') if d.strip()]

        if len(smi_list) != len(doses):
            raise ValueError(f"Number of SMILES ({len(smi_list)}) does not match number of doses ({len(doses)}).")
    
        print('\033[93mPlease select dose unit [mg or mg/kg]:\033[0m')
        dose_unit = input('> ').strip().lower()
        if dose_unit not in ['mg', 'mg/kg']:
            raise ValueError(f"Invalid dose unit: '{dose_unit}'. Must be 'mg' or 'mg/kg'.")
    
        print('\033[93mDo you want to standardize SMILES? [y/n] (default: y):\033[0m')
        std_input = input('> ').strip().lower()
        if std_input in ['', 'y', 'yes']:
            standardize_smi = True
        elif std_input in ['n', 'no']:
            standardize_smi = False
        else:
            raise ValueError("Invalid input for standardization. Please enter 'y' or 'n'.")

    elif input_mode == '2':
        print('\033[93mPlease enter the path to your CSV file:\033[0m')
        csv_path = Path('examples', input('> ').strip())
        df = pd.read_csv(csv_path)
        assert 'smiles' in df.columns and 'dose' in df.columns, \
            "CSV file must contain 'smiles' and 'dose' columns"
        smi_list = df['smiles'].astype(str).tolist()
        doses = df['dose'].astype(float).tolist()
        print('\033[93mPlease select dose unit [mg or mg/kg]:\033[0m')
        dose_unit = input('> ').strip().lower()
        if dose_unit not in ['mg', 'mg/kg']:
            raise ValueError(f"Invalid dose unit: '{dose_unit}'. Must be 'mg' or 'mg/kg'.")
    
        print('\033[93mDo you want to standardize SMILES? [y/n] (default: y):\033[0m')
        std_input = input('> ').strip().lower()
        if std_input in ['', 'y', 'yes']:
            standardize_smi = True
        elif std_input in ['n', 'no']:
            standardize_smi = False
        else:
            raise ValueError("Invalid input for standardization. Please enter 'y' or 'n'.")
    else:
        raise ValueError("Invalid selection. Please enter 1 or 2.")
    
    print('\n===== Input Summary =====')
    
    for i, (smi, dose) in enumerate(zip(smi_list, doses)):
        print(f"Compound \033[92m{i+1}\033[0m: SMILES = \033[92m{smi}\033[0m, Dose = \033[92m{dose} {dose_unit}\033[0m")
    print(f"SMILES Standardization: \033[92m{'Enabled' if standardize_smi else 'Disabled'}\033[0m")
    print('==========================\n')
    print('Proceeding in 5 seconds... (Ctrl+C to cancel)')
    time.sleep(5)

    return smi_list, doses, dose_unit, standardize_smi

if __name__ == '__main__':
    seed_everything(args.seed)
    device = torch.device(
        'cuda:' + str(args.device)
        if torch.cuda.is_available() else 'cpu'
    )
    num_tasks = len(args.pk_params)
    
    # Input
    smi_list, doses, dose_unit, standardize_smi = get_user_input()

    df_preds_by_task = {task: [] for task in TASKS}
    att_pred = []
    
    for fold in range(1, 11):
        model = MMPKPredictor(args, num_tasks=num_tasks)
        model.to(device)
    
        model_path = f"checkpoints/{args.checkpoints_folder}/fold_{fold}.pth"
        print(f"Loading MMPK model {fold} from checkpoints/{args.checkpoints_folder} for new prediction...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        loader = MMPKPredictLoader(smi_list, doses, dose_unit, standardize_smi).get_loader()
        
        preds, att_info = test(model, loader, device)
    
        for task in TASKS:
            df_preds = pd.DataFrame(preds[task])
            df_preds_by_task[task].append(df_preds)
        
        for info in att_info:
            for sub in info['sub_scores']:
                att_pred.append({
                    'fold': fold,
                    'smiles': info['smiles'],
                    'dose': info['dose'],
                    'sub_index': sub['sub_index'],
                    'sub_smiles': sub['sub_smiles'],
                    'sub_score': sub['sub_score']
                })
    
    # Prediction
    ref_df = df_preds_by_task[TASKS[0]][0][['smiles', 'dose']].reset_index(drop=True)
    pred_df = ref_df.copy()
    
    for task in TASKS:
        print(f"Processing {task}...")
        dfs = df_preds_by_task[task]
        
        merged_df = pd.concat([
            df[['y_hat']].rename(columns={'y_hat': f"y_hat_fold_{i+1}"})
            for i, df in enumerate(dfs)
        ], axis=1)
        
        pred_df[f"{task}_log"] = merged_df.mean(axis=1)
        y_hat_orig = back_transform_predict(y_hat_log=pred_df[f"{task}_log"], pk_param=task)
        pred_df[task] = y_hat_orig.round(3)
    
    # show log and original prediction
    task_cols = TASKS
    log_cols = [f"{task}_log" for task in TASKS]
    ordered_cols = ['smiles', 'dose'] + task_cols + log_cols
    pred_df = pred_df[ordered_cols]
    unit_mapping = {
        'smiles': 'SMILES',
        'dose': f"Dose [{dose_unit}]",
        'auc': 'AUC [ng*h/mL]',
        'cmax': 'Cmax [ng/mL]',
        'tmax': 'Tmax [h]',
        'hl': 't1/2 [h]',
        'cl': 'CL/F [L/h]',
        'vz': 'Vz/F [L]',
        'mrt': 'MRT [h]',
        'f': 'F [%]',
        'auc_log': 'AUC [log]',
        'cmax_log': 'Cmax [log]',
        'tmax_log': 'Tmax [log]',
        'hl_log': 't1/2 [log]',
        'cl_log': 'CL/F [log]',
        'vz_log': 'Vz/F [log]',
        'mrt_log': 'MRT [log]',
        'f_log': 'F [logit]'
    }
    pred_df = pred_df.rename(columns=unit_mapping)
    
    # Save predictions
    output_dir_pred = 'prediction'
    if not os.path.exists(output_dir_pred):
        os.makedirs(output_dir_pred)
    output_file_pred = args.output_file_pred if hasattr(args, 'output_file_pred') else 'prediction.csv'
    print(f"\033[92m[SAVE] Saving predictions to {output_dir_pred}/{output_file_pred}\033[0m")
    pred_df.to_csv(os.path.join(output_dir_pred, output_file_pred), index=False)
    
    # Attention
    df_att = pd.DataFrame(att_pred)
    df_att_mean = (
        df_att
        .groupby(['smiles', 'dose', 'sub_index', 'sub_smiles'], as_index=False)
        ['sub_score']
        .mean()
    )
    df_att_mean['sub_score'] = df_att_mean['sub_score'].round(3)
    
    att_col_mapping = {
        'smiles': 'SMILES',
        'dose': f"Dose [{dose_unit}]",
        'sub_index': 'Substructure Index',
        'sub_smiles': 'Substructure SMILES',
        'sub_score': 'Attention Weight'
    }
    df_att_mean = df_att_mean.rename(columns=att_col_mapping)
    
    # Save attention scores
    output_dir_att = 'attention'
    if not os.path.exists(output_dir_att):
        os.makedirs(output_dir_att)
    output_file_att = args.output_file_att if hasattr(args, 'output_file_att') else 'attention.csv'
    print(f"\033[92m[SAVE] Saving attention weights to {output_dir_att}/{output_file_att}\033[0m")
    df_att_mean.to_csv(os.path.join(output_dir_att, output_file_att), index=False)
    