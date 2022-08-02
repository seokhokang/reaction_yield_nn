import os, csv
import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

def mol_dict():
    return {'n_node': [],
            'n_edge': [],
            'node_attr': [],
            'edge_attr': [],
            'src': [],
            'dst': []}

def get_graph_data(rsmi_list, yld_list, data_id, filename):

    def add_mol(mol_dict, mol):

        def _DA(mol):
    
            D_list, A_list = [], []
            for feat in chem_feature_factory.GetFeaturesForMol(mol):
                if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
                if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
            
            return D_list, A_list

        def _chirality(atom):
            
            if atom.HasProp('Chirality'):
                c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
            else:
                c_list = [0, 0]

            return c_list
            
        def _stereochemistry(bond):
            
            if bond.HasProp('Stereochemistry'):
                s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
            else:
                s_list = [0, 0]

            return s_list     
            

        n_node = mol.GetNumAtoms()
        n_edge = mol.GetNumBonds() * 2
        
        D_list, A_list = _DA(mol)  
        atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-1]
        atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
        atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_fea8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        atom_fea10 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
        
        node_attr = np.hstack([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10])
    
        mol_dict['n_node'].append(n_node)
        mol_dict['n_edge'].append(n_edge)
        mol_dict['node_attr'].append(node_attr)
    
        if n_edge > 0:

            bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
            bond_fea2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
            bond_fea3 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
            
            edge_attr = np.hstack([bond_fea1, bond_fea2, bond_fea3])
            edge_attr = np.vstack([edge_attr, edge_attr])
            
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int)
            src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
            dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
        
            mol_dict['edge_attr'].append(edge_attr)
            mol_dict['src'].append(src)
            mol_dict['dst'].append(dst)
        
        return mol_dict

    def add_dummy(mol_dict):

        n_node = 1
        n_edge = 0
        node_attr = np.zeros((1, len(atom_list) + len(charge_list) + len(degree_list) + len(hybridization_list) + len(hydrogen_list) + len(valence_list) + len(ringsize_list) + 1))
    
        mol_dict['n_node'].append(n_node)
        mol_dict['n_edge'].append(n_edge)
        mol_dict['node_attr'].append(node_attr)
        
        return mol_dict
  
    def dict_list_to_numpy(mol_dict):
    
        mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
        mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
        mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
        if np.sum(mol_dict['n_edge']) > 0:
            mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
            mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
            mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
        else:
            mol_dict['edge_attr'] = np.empty((0, len(bond_list) + 2)).astype(bool)
            mol_dict['src'] = np.empty(0).astype(int)
            mol_dict['dst'] = np.empty(0).astype(int)

        return mol_dict
    
    if data_id == 1: atom_list = ['C','N','O','F','P','S','Cl','Br','Pd','I']
    elif data_id == 2: atom_list = ['Li','B','C','N','O','F','Na','P','S','Cl','K','Fe','Br','Pd','I','Cs']
    
    charge_list = [1, 2, -1, 0]
    degree_list = [1, 2, 3, 4, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S']
    hydrogen_list = [1, 2, 3, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    
    rmol_max_cnt = np.max([smi.split('>>')[0].count('.') + 1 for smi in rsmi_list])
    pmol_max_cnt = np.max([smi.split('>>')[1].count('.') + 1 for smi in rsmi_list])

    rmol_dict = [mol_dict() for _ in range(rmol_max_cnt)]
    pmol_dict = [mol_dict() for _ in range(pmol_max_cnt)]
                 
    reaction_dict = {'yld': [], 'rsmi': []}     
    
    print('--- generating graph data for %s' %filename)
    print('--- n_reactions: %d, reactant_max_cnt: %d, product_max_cnt: %d' %(len(rsmi_list), rmol_max_cnt, pmol_max_cnt)) 
                 
    for i in range(len(rsmi_list)):
    
        rsmi = rsmi_list[i].replace('~', '-')
        yld = yld_list[i]
    
        [reactants_smi, products_smi] = rsmi.split('>>')
        
        # processing reactants
        reactants_smi_list = reactants_smi.split('.')
        for _ in range(rmol_max_cnt - len(reactants_smi_list)): reactants_smi_list.append('')
        for j, smi in enumerate(reactants_smi_list):
            if smi == '':
                rmol_dict[j] = add_dummy(rmol_dict[j]) 
            else:
                rmol = Chem.MolFromSmiles(smi)
                rs = Chem.FindPotentialStereo(rmol)
                for element in rs:
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': rmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': rmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))

                rmol = Chem.RemoveHs(rmol)
                rmol_dict[j] = add_mol(rmol_dict[j], rmol) 
        
        # processing products
        products_smi_list = products_smi.split('.')
        for _ in range(pmol_max_cnt - len(products_smi_list)): products_smi_list.append('') 
        for j, smi in enumerate(products_smi_list):
            if smi == '':
                pmol_dict[j] = add_dummy(pmol_dict[j])
            else: 
                pmol = Chem.MolFromSmiles(smi)
                ps = Chem.FindPotentialStereo(pmol)
                for element in ps:
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': pmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': pmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                        
                pmol = Chem.RemoveHs(pmol) 
                pmol_dict[j] = add_mol(pmol_dict[j], pmol)  
        
        # yield and reaction SMILES
        reaction_dict['yld'].append(yld)
        reaction_dict['rsmi'].append(rsmi)
    
        # monitoring
        if (i+1) % 1000 == 0: print('--- %d/%d processed' %(i+1, len(rsmi_list))) 
        
    # datatype to numpy
    for j in range(rmol_max_cnt): rmol_dict[j] = dict_list_to_numpy(rmol_dict[j])   
    for j in range(pmol_max_cnt): pmol_dict[j] = dict_list_to_numpy(pmol_dict[j])   
    reaction_dict['yld'] = np.array(reaction_dict['yld'])
    
    # save file
    np.savez_compressed(filename, data = [rmol_dict, pmol_dict, reaction_dict]) 

    
if __name__ == "__main__":

    for data_id in [1, 2]:
        for split_id in range(10):

            load_dict = np.load('./split/data%d_split_%d.npz' %(data_id, split_id), allow_pickle = True)
            
            rsmi_list = load_dict['data_df'][:,0]
            yld_list = load_dict['data_df'][:,1]
            filename = './dataset_%d_%d.npz' %(data_id, split_id)
            
            get_graph_data(rsmi_list, yld_list, data_id, filename)

    for test_id in [1, 2, 3, 4]:

        data_id = 1

        load_dict = np.load('./split/Test%d_split.npz' %test_id, allow_pickle = True)

        rsmi_list = load_dict['data_df'][:,0]
        yld_list = load_dict['data_df'][:,1]
        filename = './test_%d.npz' %test_id
    
        get_graph_data(rsmi_list, yld_list, data_id, filename)
