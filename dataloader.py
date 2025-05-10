import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from mmpk import MMPKDataset, MMPKPredictDataset
from utils import standardize
from config import args

def load_data(n_fold, pk_params):
    print(f"Loading training data - Fold: {n_fold} - PK params: {', '.join(pk_params)} - Number of tasks: {len(pk_params)}")
    
    train_data = pd.read_csv(f"data/approved/fold{n_fold}/train_data_fold_{n_fold}.csv")
    val_data = pd.read_csv(f"data/approved/fold{n_fold}/val_data_fold_{n_fold}.csv")
    test_data = pd.read_csv(f"data/approved/fold{n_fold}/test_data_fold_{n_fold}.csv")

    X_train, y_train, dose_train = train_data['smiles'].tolist(), train_data[[f'log_{param}' for param in pk_params]].values, train_data['log_dose'].values
    X_val, y_val, dose_val = val_data['smiles'].tolist(), val_data[[f'log_{param}' for param in pk_params]].values, val_data['log_dose'].values
    X_test, y_test, dose_test = test_data['smiles'].tolist(), test_data[[f'log_{param}' for param in pk_params]].values, test_data['log_dose'].values

    return (X_train, y_train, dose_train), (X_val, y_val, dose_val), (X_test, y_test, dose_test)

def load_ext_data(pk_params):
    print(f"Loading external data - PK Params: {', '.join(pk_params)} - Number of tasks: {len(pk_params)}")
    
    ext_invest_data = pd.read_csv('data/investigational/investigational_modeling.csv')
    ext_2024_data = pd.read_csv('data/approved_2024/approved_2024_modeling.csv')

    X_ext_invest, y_ext_invest, dose_ext_invest = ext_invest_data['smiles'].tolist(), ext_invest_data[[f'log_{param}' for param in pk_params]].values, ext_invest_data['log_dose'].values
    X_ext_2024, y_ext_2024, dose_ext_2024 = ext_2024_data['smiles'].tolist(), ext_2024_data[[f'log_{param}' for param in pk_params]].values, ext_2024_data['log_dose'].values

    return (X_ext_invest, y_ext_invest, dose_ext_invest), (X_ext_2024, y_ext_2024, dose_ext_2024)

def mmpk_collate_fn(batch, dataset):
    mol_gdata, sub_mask, token_ids, token_mask, label, dose = zip(*batch)

    mol_gdata = Batch.from_data_list(mol_gdata)
    sub_gdata = Batch.from_data_list(dataset.sub_gdata_list)

    return mol_gdata, sub_gdata, torch.stack(sub_mask), torch.stack(token_ids), torch.stack(token_mask), torch.stack(label), torch.stack(dose)

def get_mmpk_loader(train_data, val_data, test_data):
    train_dataset = MMPKDataset(*train_data)
    val_dataset = MMPKDataset(*val_data)
    test_dataset = MMPKDataset(*test_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda batch: mmpk_collate_fn(batch, train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda batch: mmpk_collate_fn(batch, val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda batch: mmpk_collate_fn(batch, test_dataset))

    return train_loader, val_loader, test_loader

def get_ext_loader(ext_invest_data, ext_2024_data):
    ext_invest_dataset = MMPKDataset(*ext_invest_data)
    ext_2024_dataset = MMPKDataset(*ext_2024_data)

    ext_invest_loader = DataLoader(ext_invest_dataset, batch_size=len(ext_invest_dataset), shuffle=False,
                              collate_fn=lambda batch: mmpk_collate_fn(batch, ext_invest_dataset))
    ext_2024_loader = DataLoader(ext_2024_dataset, batch_size=len(ext_2024_dataset), shuffle=False,
                              collate_fn=lambda batch: mmpk_collate_fn(batch, ext_2024_dataset))

    return ext_invest_loader, ext_2024_loader

class DataLoaderSelector:
    def __init__(self, n_fold, pk_params, load_ext=False):
        self.n_fold = n_fold
        self.pk_params = pk_params
        self.load_ext = load_ext

        if self.load_ext:
            self.ext_data = self.load_ext_data()
            self.data = None
        else:
            self.data = self.load_data()
            self.ext_data = None
        
    def load_data(self):
        return load_data(self.n_fold, self.pk_params)
    
    def load_ext_data(self):
        return load_ext_data(self.pk_params)
    
    def select_loader(self, loader_type):
        if loader_type == 'mmpk':
            return get_mmpk_loader(*self.data)
        elif loader_type == 'ext':
            return get_ext_loader(*self.ext_data)

####### Predict Loader #######

def predict_collate_fn(batch, dataset):
    mol_gdata, sub_mask, token_ids, token_mask, dose, orig_dose = zip(*batch)

    mol_gdata = Batch.from_data_list(mol_gdata)
    sub_gdata = Batch.from_data_list(dataset.sub_gdata_list)

    return mol_gdata, sub_gdata, torch.stack(sub_mask), torch.stack(token_ids), torch.stack(token_mask), torch.stack(dose), torch.stack(orig_dose)

class MMPKPredictLoader:
    def __init__(self, smi_list, doses, dose_unit='mg/kg', standardize_smi=True):
        self.standardize = standardize_smi
        if self.standardize:
            smi_list = [standardize(smi) for smi in smi_list]
        self.dataset = MMPKPredictDataset(smi_list, doses, dose_unit)
    
    def get_loader(self, batch_size=None):
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size or len(self.dataset),
            shuffle=False,
            collate_fn=lambda batch: predict_collate_fn(batch, self.dataset)
        )
        return loader

if __name__ == '__main__':
    pass
