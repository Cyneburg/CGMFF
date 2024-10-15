import os
from itertools import islice
import sys
import random
import numpy as np
from rdkit import Chem
import networkx as nx
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.data import Data
import pandas as pd
import torch
from splitters import *
from model.Mol_BERT.MOLEBert import *
from creat_data_DC import creat_data
import warnings
warnings.filterwarnings("ignore")

class MPPDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, y=None, smile_graph=None, clique_graph=None):
        super(MPPDataset, self).__init__(root, )
        self.dataset = dataset
        self.y = y
        self.smile_graph = smile_graph
        self.clique_graph = clique_graph
        self.process(xd, y, smile_graph, clique_graph)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_clique.pt', self.dataset + '_data_pre.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, y, smile_graph, clique_graph):
        data_list_mol = []
        data_list_clique = []
        data_list_pre = []
        data_len = len(xd)
        # print('loading tensors ...')
        for i in range(data_len):
            # print('Converting to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            labels = y[i]
            mol_size, mol_features, mol_edge_index, edge_features_list = smile_graph[smiles]
            clique_size, clique_features, clique_edge_index = clique_graph[smiles]
            # print(labels)
            # pre_mol =  mol_to_graph_data_obj_simple(mol)
            GCNData_mol = Data(x=torch.Tensor(mol_features),
                               edge_index=torch.LongTensor(mol_edge_index).transpose(-1, 0),
                               edge_attr= torch.LongTensor(edge_features_list),
                               y=torch.FloatTensor([labels]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            GCNData_clique = Data(x=torch.Tensor(clique_features),
                                  edge_index=torch.LongTensor(clique_edge_index).transpose(-1, 0),
                                  y=torch.FloatTensor([labels]))
            GCNData_clique.__setitem__('target_size', torch.LongTensor([clique_size]))

            data_list_mol.append(GCNData_mol)
            data_list_clique.append(GCNData_clique)
            # data_list_pre.append(pre_mol)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_clique = [data for data in data_list_clique if self.pre_filter(data)]
            # data_list_pre = [data for data in data_list_pre if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_clique = [self.pre_transform(data) for data in data_list_clique]
            # data_list_pre = [self.pre_transform(data) for data in data_list_pre]

        self.data_mol = data_list_mol
        self.data_clique = data_list_clique
        # self.data_pre = data_list_pre

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
        return self.data_mol[idx], self.data_clique[idx] #, self.data_pre[idx]

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def encoding_unk(x, allowable_set):
    list = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            list[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        list[-1] = True
    return list


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_size = mol.GetNumAtoms()
    mol_features = []
    edge_index, edge_features_list= [], []
    app_atoms = []

    for bond in mol.GetBonds():
        app_atoms.append(bond.GetBeginAtomIdx())
        app_atoms.append(bond.GetEndAtomIdx())
        edge_feature = [allowable_features['possible_bonds'].index(
            bond.GetBondType())] + [allowable_features['possible_bond_dirs'].index(
            bond.GetBondDir())]
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_features_list.append(edge_feature)
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_features_list.append(edge_feature)

    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        mol_features.append(feature / sum(feature))

    return mol_size, mol_features, edge_index, edge_features_list

def clique_features(clique, edges, clique_idx, smile):
    NumAtoms = len(clique)  # 子结构中原子数
    NumEdges = 0  # 与子结构所连的边数，子结构的度
    for edge in edges:
        if clique_idx == edge[0] or clique_idx == edge[1]:
            NumEdges += 1
    mol = Chem.MolFromSmiles(smile)
    atoms = []
    NumHs = 0  # 基团中氢原子的个数
    NumImplicitValence = 0
    for idx in clique:
        atom = mol.GetAtomWithIdx(idx)
        atoms.append(atom.GetSymbol())
        NumHs += atom.GetTotalNumHs()
        NumImplicitValence += atom.GetImplicitValence()
    # 基团中是否包含环
    IsRing = 0
    if len(clique) > 2:
        IsRing = 1
    # 基团中是否有键
    IsBond = 0
    if len(clique) == 2:
        IsBond = 1
    return np.array(encoding_unk(atoms, ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding_unk(NumAtoms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding_unk(NumEdges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding_unk(NumHs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) +
                    one_hot_encoding_unk(NumImplicitValence, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) +
                    [IsRing]+[IsBond])

def cluster_graph(mol):
    n_atoms = mol.GetNumAtoms()  #获得分子图中原子数
    cliques = []  # 所有的的边（化学键和一对原子组成）和环
    for bond in mol.GetBonds():  #获得分子图中所有化学键
        a1 = bond.GetBeginAtom().GetIdx() #获得开始原子索引
        a2 = bond.GetEndAtom().GetIdx() #获得结束原子索引
        if not bond.IsInRing():   #判断化学键是否在环中
            cliques.append([a1, a2])  #将边加入到集合中

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]  # 获得分子图中的所有环
    cliques.extend(ssr)  #
    # cliques 所有的的边（化学键和一对原子组成）和环
    # nei_list为原子属于哪个基团\子结构
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    edges = []
    for i in range(len(cliques)-1):
        for j in range(i+1,len(cliques)):
            if len(set(cliques[i]) & set(cliques[j]))!= 0:
                edges.append([i,j])
                edges.append([j,i])
    return cliques, edges

def clique_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    clique, edge = cluster_graph(mol)

    c_features = []  # 特征矩阵
    for idx in range(len(clique)):
        cq_features = clique_features(clique[idx], edge, idx, smile)
        c_features.append(cq_features / sum(cq_features))

    clique_size = len(clique)  # 子结构图节点数
    return clique_size, c_features, edge

def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data[0].y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    # print(dataset,train_idx)
    train_dataset = [dataset[id] for id in train_idx]
    valid_dataset = [dataset[id] for id in valid_idx]
    test_dataset = [dataset[id] for id in test_idx]
    return train_dataset, valid_dataset, test_dataset

def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = [dataset[id] for id in train_idx]
    valid_dataset = [dataset[id] for id in valid_idx]
    test_dataset = [dataset[id] for id in test_idx]

    return train_dataset, valid_dataset, test_dataset

def random_split(dataset, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0,
                 smiles_list=None):
    """

    :param dataset:
    :param task_idx:
    :param null_value:
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed:
    :param smiles_list: list of smiles corresponding to the dataset obj, or None
    :return: train, valid, test slices of the input dataset obj. If
    smiles_list != None, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value  # boolean array that correspond to non null values
        idx_array = np.where(non_null)[0]
        dataset = dataset[torch.tensor(idx_array)]  # examples containing non
        # null labels in the specified task_idx
    else:
        pass

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = [dataset[id] for id in train_idx]
    valid_dataset = [dataset[id] for id in valid_idx]
    test_dataset = [dataset[id] for id in test_idx]

    return train_dataset, valid_dataset, test_dataset


def creat_data(datasets, mode, seed):
    if datasets == 'baidu':
        compound_iso_smiles = []
        if mode == 'train':
            df = pd.read_csv('data/baidu/train.csv')
            compound_iso_smiles += list(df['smiles'])
            compound_iso_smiles = set(compound_iso_smiles)
            smile_graph = {}
            clique_graph = {}
            print('compound_iso_smiles', compound_iso_smiles)
            for smile in compound_iso_smiles:
                g = smile_to_graph(smile)
                clique = clique_to_graph(smile)
                smile_graph[smile] = g
                clique_graph[smile] = clique
                # convert to PyTorch data format
            random.seed(seed)
            df = pd.read_csv('data/baidu/train.csv')
            random_num = random.sample(range(0, len(df)), len(df))
            pot = len(df) // 5
            valid_num = random_num[pot:pot * 2]
            train_num = random_num[:pot] + random_num[pot * 2:]
            data_train = df.loc[train_num]
            data_valid = df.loc[valid_num]
            drug_train, label_train, drug_valid, label_valid = list(data_train['smiles']), list(data_train['label']),list(data_valid['smiles']), list(data_valid['label'])
            drug_train, label_train, drug_valid, label_valid = np.asarray(drug_train), np.asarray(label_train),np.asarray(drug_valid), np.asarray(label_valid)
            # make data PyTorch Geometric ready
            print('开始创建数据')
            print(datasets)
            train_data = MPPDataset(root='data', dataset=datasets, xd=drug_train, y=label_train, smile_graph=smile_graph, clique_graph=clique_graph)
            valid_data = MPPDataset(root='data', dataset=datasets, xd=drug_valid, y=label_valid, smile_graph=smile_graph, clique_graph=clique_graph)
            print('创建数据成功')

            return train_data,valid_data
        else:
            df = pd.read_csv('data/baidu/test.csv'.format(datasets))
            compound_iso_smiles += list(df['smiles'])
            compound_iso_smiles = set(compound_iso_smiles)
            smile_graph = {}
            clique_graph = {}
            print('compound_iso_smiles', compound_iso_smiles)
            for smile in compound_iso_smiles:
                g = smile_to_graph(smile)
                clique = clique_to_graph(smile)
                smile_graph[smile] = g
                clique_graph[smile] = clique
            df = pd.read_csv('data/' + datasets + '/test.csv')
            df['label'] = [0] * len(df)
            drug, label = list(df['smiles']), list(df['label'])
            drug, label = np.asarray(drug), np.asarray(label)
            dataset = MPPDataset(root='data', dataset=datasets, xd=drug, y=label, smile_graph=smile_graph,
                                 clique_graph=clique_graph)
            print('创建数据成功')
            return dataset
    elif datasets == 'tox21':
        input_df = pd.read_csv('data/MPP/tox21/raw/tox21.csv', sep=',')
        smiles_list = input_df['smiles']
        tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        labels = input_df[tasks]
        labels = labels.replace(0, -1)
        labels = labels.fillna(0)
        smile_graph = {}
        clique_graph = {}
        for smile in smiles_list:
            g = smile_to_graph(smile)
            clique = clique_to_graph(smile)
            smile_graph[smile] = g
            clique_graph[smile] = clique
        assert len(smiles_list) == len(smile_graph)
        assert len(smiles_list) == len(clique_graph)
        drug = list(smiles_list)
        label = list(labels.values)
        print('开始创建数据')
        print(datasets)
        data = MPPDataset(root='data', dataset=datasets, xd=drug, y=label, smile_graph=smile_graph,clique_graph=clique_graph)
        print('创建数据成功')
        return data, smiles_list

    elif datasets == 'sider':
        input_df = pd.read_csv('data/MPP/sider/raw/sider.csv', sep=',')
        smiles_list, label = [], []
        tasks = ['Hepatobiliary disorders',
       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
       'Investigations', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Social circumstances',
       'Immune system disorders', 'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications']
        labels = input_df[tasks]
        labels = labels.replace(0, -1)
        smile_graph = {}
        clique_graph = {}
        for i in range(len(input_df)):
            try:
                g = smile_to_graph(input_df.loc[i]['smiles'])
                clique = clique_to_graph(input_df.loc[i]['smiles'])
                smile_graph[input_df.loc[i]['smiles']] = g
                clique_graph[input_df.loc[i]['smiles']] = clique
                smiles_list.append(input_df.loc[i]['smiles'])
                label.append(labels.loc[i])
            except:
                pass
        drug = smiles_list
        print('开始创建数据')
        print(datasets)
        data = MPPDataset(root='data', dataset=datasets, xd=drug, y=label, smile_graph=smile_graph,
                          clique_graph=clique_graph)
        print('创建数据成功')
        return data, smiles_list

    elif datasets == 'clintox':
        input_df = pd.read_csv('data/MPP/clintox/raw/clintox.csv', sep=',')
        smiles_list, label = [], []
        tasks = ['FDA_APPROVED', 'CT_TOX']
        labels = input_df[tasks]
        labels = labels.replace(0, -1)
        smile_graph = {}
        clique_graph = {}
        for i in range(len(input_df)):
            try:
                g = smile_to_graph(input_df.loc[i]['smiles'])
                clique = clique_to_graph(input_df.loc[i]['smiles'])
                smile_graph[input_df.loc[i]['smiles']] = g
                clique_graph[input_df.loc[i]['smiles']] = clique
                smiles_list.append(input_df.loc[i]['smiles'])
                label.append(labels.loc[i])
            except:
                pass
        drug = smiles_list
        print('开始创建数据')
        print(datasets)
        data = MPPDataset(root='data', dataset=datasets, xd=drug, y=label, smile_graph=smile_graph,
                          clique_graph=clique_graph)
        print('创建数据成功')
        return data, smiles_list

    elif datasets == 'bbbp':
        input_df = pd.read_csv('data/MPP/bbbp/raw/BBBP.csv', sep=',')
        smiles_list, label = [], []
        smile_graph = {}
        clique_graph = {}
        for i in range(len(input_df)):
            try:
                g = smile_to_graph(input_df.loc[i]['smiles'])
                clique = clique_to_graph(input_df.loc[i]['smiles'])
                smile_graph[input_df.loc[i]['smiles']] = g
                clique_graph[input_df.loc[i]['smiles']] = clique
                smiles_list.append(input_df.loc[i]['smiles'])
                label.append(input_df.loc[i]['p_np'])
            except:
                pass
        drug = smiles_list
        print('开始创建数据')
        print(datasets)
        data = MPPDataset(root='data', dataset=datasets, xd=drug, y=label, smile_graph=smile_graph,
                          clique_graph=clique_graph)
        print('创建数据成功')
        return data, smiles_list

    elif datasets == 'bace':
        input_df = pd.read_csv('data/MPP/bace/raw/bace.csv', sep=',')
        smiles_list, label = [], []
        smile_graph = {}
        clique_graph = {}
        for i in range(len(input_df)):
            try:
                g = smile_to_graph(input_df.loc[i]['mol'])
                clique = clique_to_graph(input_df.loc[i]['mol'])
                smile_graph[input_df.loc[i]['mol']] = g
                clique_graph[input_df.loc[i]['mol']] = clique
                smiles_list.append(input_df.loc[i]['mol'])
                label.append(input_df.loc[i]['Class'])
            except:
                pass
        drug = smiles_list
        print('开始创建数据')
        print(datasets)
        data = MPPDataset(root='data', dataset=datasets, xd=drug, y=label, smile_graph=smile_graph,
                          clique_graph=clique_graph)
        print('创建数据成功')
        return data, smiles_list

    elif datasets == 'esol':
        random.seed(seed)
        input_df = pd.read_csv('data/MPP/esol/raw/delaney-processed.csv', sep=',')
        smiles_list, label = [], []
        smile_graph = {}
        clique_graph = {}
        for i in range(len(input_df)):
            try:
                g = smile_to_graph(input_df.loc[i]['smiles'])
                clique = clique_to_graph(input_df.loc[i]['smiles'])
                smile_graph[input_df.loc[i]['smiles']] = g
                clique_graph[input_df.loc[i]['smiles']] = clique
                smiles_list.append(input_df.loc[i]['smiles'])
                label.append(input_df.loc[i]['measured log solubility in mols per litre'])
            except:
                pass
        drug = smiles_list
        print('开始创建数据')
        print(datasets)
        data = MPPDataset(root='data', dataset=datasets, xd=drug, y=label, smile_graph=smile_graph,
                          clique_graph=clique_graph)
        print('创建数据成功')
        return data, smiles_list

    elif datasets == 'lipo':
        random.seed(seed)
        input_df = pd.read_csv('data/MPP/lipophilicity/raw/Lipophilicity.csv', sep=',')
        smiles_list, label = [], []
        smile_graph = {}
        clique_graph = {}
        for i in range(len(input_df)):
            try:
                g = smile_to_graph(input_df.loc[i]['smiles'])
                clique = clique_to_graph(input_df.loc[i]['smiles'])
                smile_graph[input_df.loc[i]['smiles']] = g
                clique_graph[input_df.loc[i]['smiles']] = clique
                smiles_list.append(input_df.loc[i]['smiles'])
                label.append(input_df.loc[i]['exp'])
            except:
                pass
        drug = smiles_list
        print('开始创建数据')
        print(datasets)
        data = MPPDataset(root='data', dataset=datasets, xd=drug, y=label, smile_graph=smile_graph,
                          clique_graph=clique_graph)
        print('创建数据成功')
        return data, smiles_list