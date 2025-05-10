from rdkit import Chem
from rdkit.Chem import BRICS, Recap

def get_substructure(smi, decomp='brics') -> list:
    mol = Chem.MolFromSmiles(smi)

    assert decomp in ['brics', 'recap']

    if decomp == 'brics':
        substructures = BRICS.BRICSDecompose(mol)
    elif decomp == 'recap':
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        substructures = set(leaves.keys())

    return list(substructures)

if __name__ == '__main__':
    pass
