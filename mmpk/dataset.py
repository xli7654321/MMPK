import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from tqdm import tqdm
from .brics import get_substructure
from gnn import get_graph_data

class MMPKDataset(Dataset):
    def __init__(self, smi_list, labels, doses):
        super().__init__()
        tokenizer = RobertaTokenizer.from_pretrained('ChemBERTa/ChemBERTa-77M-MTR')
        inputs = tokenizer(smi_list, return_tensors='pt', padding=True)
        self.token_ids = inputs['input_ids']
        self.token_mask = inputs['attention_mask']

        self.mol_gdata_list = [get_graph_data(smi) for smi in tqdm(smi_list, desc='Generating Mol Graph Data')]
        subs_list = [get_substructure(smi) for smi in tqdm(smi_list, desc='Generating Substructure by BRICS')]
        uni_sub_list = sorted(set().union(*subs_list))
        self.sub_to_idx = {x: idx for idx, x in enumerate(uni_sub_list)}
        self.sub_gdata_list = [get_graph_data(sub) for sub in tqdm(uni_sub_list, desc='Generating Sub Graph Data')]
        
        # mask is used to indicate which substructures belong to the current molecule
        # mask.shape: (num_mols, num_subs)
        self.sub_mask = torch.zeros((len(smi_list), len(uni_sub_list)), dtype=torch.bool)

        for mol_idx, subs in enumerate(subs_list):
            self.sub_mask[mol_idx, [self.sub_to_idx[sub] for sub in subs]] = True

        # mask = 0 if sub is not in the mol, mask = 1 if sub is in the mol
        self.sub_mask = self.sub_mask.long()

        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.doses = torch.tensor(doses, dtype=torch.float32)

    def __len__(self):
        return len(self.mol_gdata_list)

    def __getitem__(self, idx):
        return (
            self.mol_gdata_list[idx],
            self.sub_mask[idx],
            self.token_ids[idx],
            self.token_mask[idx],
            self.labels[idx],
            self.doses[idx]
        )

class MMPKPredictDataset(Dataset):
    def __init__(self, smi_list, doses, dose_unit='mg/kg'):
        super().__init__()
        tokenizer = RobertaTokenizer.from_pretrained('ChemBERTa/ChemBERTa-77M-MTR')
        inputs = tokenizer(smi_list, return_tensors='pt', padding=True)
        self.token_ids = inputs['input_ids']
        self.token_mask = inputs['attention_mask']

        self.mol_gdata_list = [get_graph_data(smi) for smi in tqdm(smi_list, desc='Generating Mol Graph Data')]
        subs_list = [get_substructure(smi) for smi in tqdm(smi_list, desc='Generating Substructure by BRICS')]
        uni_sub_list = sorted(set().union(*subs_list))
        self.sub_to_idx = {x: idx for idx, x in enumerate(uni_sub_list)}
        self.sub_gdata_list = [get_graph_data(sub) for sub in tqdm(uni_sub_list, desc='Generating Sub Graph Data')]
        
        # mask is used to indicate which substructures belong to the current molecule
        # mask.shape: (num_mols, num_subs)
        self.sub_mask = torch.zeros((len(smi_list), len(uni_sub_list)), dtype=torch.bool)

        for mol_idx, subs in enumerate(subs_list):
            self.sub_mask[mol_idx, [self.sub_to_idx[sub] for sub in subs]] = True

        # mask = 0 if sub is not in the mol, mask = 1 if sub is in the mol
        self.sub_mask = self.sub_mask.long()
        
        # save original doses
        self.orig_doses = doses.copy()
        
        # Convert doses based on unit
        if dose_unit == 'mg':
            doses = [dose/70 for dose in doses]  # Convert to mg/kg
        elif dose_unit == 'mg/kg':
            pass
            
        # log10 transformation
        doses = np.log10(doses)
        self.doses = torch.tensor(doses, dtype=torch.float32)
        self.orig_doses = torch.tensor(self.orig_doses, dtype=torch.float32)

    def __len__(self):
        return len(self.mol_gdata_list)

    def __getitem__(self, idx):
        return (
            self.mol_gdata_list[idx],
            self.sub_mask[idx],
            self.token_ids[idx],
            self.token_mask[idx],
            self.doses[idx],
            self.orig_doses[idx]
        )

if __name__ == '__main__':
    pass
