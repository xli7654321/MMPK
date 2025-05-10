import torch
from torch_geometric.data import Data
from rdkit import Chem

allowable_atom_feats = {
    'atom_type': ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                  'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                  'K', 'Ca', 'Ga', 'Ge', 'As', 'Se', 'Br',
                  'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                  'Rb', 'Sr', 'In', 'Sn', 'Sb', 'Te', 'I',
                  'Ag', 'Cs', 'Ba', 'Bi', 'At', 'Pt', 'Au', 'Hg', 'unk'],
    'atom_degree': list(range(7)) + ['unk'],
    'atom_implicit_valence': list(range(7)) + ['unk'],
    'atom_hybridization': [Chem.rdchem.HybridizationType.SP,
                           Chem.rdchem.HybridizationType.SP2,
                           Chem.rdchem.HybridizationType.SP3,
                           Chem.rdchem.HybridizationType.SP3D,
                           Chem.rdchem.HybridizationType.SP3D2,
                           'unk'],
    'atom_formal_charge': list(range(-3, 4)) + ['unk'],
    'atom_is_aromatic': [False, True],
    'atom_is_in_ring': [False, True],
    'atom_chiral_tag': [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                        Chem.rdchem.ChiralType.CHI_OTHER,
                        'unk']
}

allowable_bond_feats = {
    'bond_type': [Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC,
                  'self_loop',
                  'unk'],
    'bond_is_conjugated': [False, True],
    'bond_is_in_ring': [False, True],
    'bond_stereo': [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE,
                    Chem.rdchem.BondStereo.STEREOCIS,
                    Chem.rdchem.BondStereo.STEREOTRANS,
                    Chem.rdchem.BondStereo.STEREOANY,
                    'unk']
}

def index_encoding(x, allowable_set):
    """
    Adapted from `ogb.utils.features.safe_index`
    If element is not in allowable_set, return the last index.
    The last element in the allowable_set needs to be 'unk'.
    """
    try:
        return allowable_set.index(x)
    except:
        return len(allowable_set) - 1

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """
    Adapted from `dgllife.utils.featurizers.one_hot_encoding`
    The last element in the allowable_set dose not need to be 'unk'.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(x == s for s in allowable_set)

def get_atom_feat_dim():
    return list(map(len, allowable_atom_feats.values()))

def get_bond_feat_dim():
    return list(map(len, allowable_bond_feats.values()))

def get_atom_feat(atom):
    atom_feat = [
        index_encoding(atom.GetSymbol(), allowable_atom_feats['atom_type']),
        index_encoding(atom.GetDegree(), allowable_atom_feats['atom_degree'] ),
        index_encoding(atom.GetImplicitValence(), allowable_atom_feats['atom_implicit_valence']),
        index_encoding(atom.GetHybridization(), allowable_atom_feats['atom_hybridization']),
        index_encoding(atom.GetFormalCharge(), allowable_atom_feats['atom_formal_charge']),
        index_encoding(atom.GetIsAromatic(), allowable_atom_feats['atom_is_aromatic']),
        index_encoding(atom.IsInRing(), allowable_atom_feats['atom_is_in_ring']),
        index_encoding(atom.GetChiralTag(), allowable_atom_feats['atom_chiral_tag'])
    ]
    
    return atom_feat

def get_bond_feat(bond):
    bond_feat = [
        index_encoding(bond.GetBondType(), allowable_bond_feats['bond_type']),
        index_encoding(bond.GetIsConjugated(), allowable_bond_feats['bond_is_conjugated']),
        index_encoding(bond.IsInRing(), allowable_bond_feats['bond_is_in_ring']),
        index_encoding(bond.GetStereo(), allowable_bond_feats['bond_stereo'])
    ]

    return bond_feat

def get_graph_data(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    
    atom_feats = []
    atom_types = []
    for atom in mol.GetAtoms():
        # The BRICS breaking point of the substructure is represented by *
        atom_feats.append(get_atom_feat(atom))
        atom_types.append(atom.GetSymbol())
    
    x = torch.tensor(atom_feats, dtype=torch.long)  # (num_atoms, num_atom_feats)

    bond_index = []
    bond_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_index.append([i, j])
        bond_feats.append(get_bond_feat(bond))
        bond_index.append([j, i])
        bond_feats.append(get_bond_feat(bond))
    
    edge_index = torch.tensor(bond_index, dtype=torch.long).t()  # (2, num_edges)
    edge_attr = torch.tensor(bond_feats, dtype=torch.long)  # (num_edges, num_edge_feats)

    gdata = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                 smi=smi, atom_types=atom_types)

    return gdata

if __name__ == '__main__':
    pass
