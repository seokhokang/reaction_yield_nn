import os, sys
import numpy as np
import torch
from dgl.convert import graph


class GraphDataset():

    def __init__(self, data_id, split_id):

        self._data_id = data_id
        self._split_id = split_id
        self.load()


    def load(self):
        
        if self._data_id in [1, 2]:
            [rmol_dict, pmol_dict, reaction_dict] = np.load('./data/dataset_%d_%d.npz' %(self._data_id, self._split_id), allow_pickle=True)['data']
        elif self._data_id == 3:
            [rmol_dict, pmol_dict, reaction_dict] = np.load('./data/test_%d.npz' %self._split_id, allow_pickle=True)['data']
    
        self.rmol_max_cnt = len(rmol_dict)
        self.pmol_max_cnt = len(pmol_dict)
    
        self.rmol_n_node = [rmol_dict[j]['n_node'] for j in range(self.rmol_max_cnt)]
        self.rmol_n_edge = [rmol_dict[j]['n_edge'] for j in range(self.rmol_max_cnt)]
        self.rmol_node_attr = [rmol_dict[j]['node_attr'] for j in range(self.rmol_max_cnt)]
        self.rmol_edge_attr = [rmol_dict[j]['edge_attr'] for j in range(self.rmol_max_cnt)]
        self.rmol_src = [rmol_dict[j]['src'] for j in range(self.rmol_max_cnt)]
        self.rmol_dst = [rmol_dict[j]['dst'] for j in range(self.rmol_max_cnt)]
        
        self.pmol_n_node = [pmol_dict[j]['n_node'] for j in range(self.pmol_max_cnt)]
        self.pmol_n_edge = [pmol_dict[j]['n_edge'] for j in range(self.pmol_max_cnt)]
        self.pmol_node_attr = [pmol_dict[j]['node_attr'] for j in range(self.pmol_max_cnt)]
        self.pmol_edge_attr = [pmol_dict[j]['edge_attr'] for j in range(self.pmol_max_cnt)]
        self.pmol_src = [pmol_dict[j]['src'] for j in range(self.pmol_max_cnt)]
        self.pmol_dst = [pmol_dict[j]['dst'] for j in range(self.pmol_max_cnt)]
        
        self.yld = reaction_dict['yld']
        self.rsmi = reaction_dict['rsmi']

        self.rmol_n_csum = [np.concatenate([[0], np.cumsum(self.rmol_n_node[j])]) for j in range(self.rmol_max_cnt)]
        self.rmol_e_csum = [np.concatenate([[0], np.cumsum(self.rmol_n_edge[j])]) for j in range(self.rmol_max_cnt)]

        self.pmol_n_csum = [np.concatenate([[0], np.cumsum(self.pmol_n_node[j])]) for j in range(self.pmol_max_cnt)]
        self.pmol_e_csum = [np.concatenate([[0], np.cumsum(self.pmol_n_edge[j])]) for j in range(self.pmol_max_cnt)]
        

    def __getitem__(self, idx):

        g1 = [graph((self.rmol_src[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]],
                     self.rmol_dst[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]]
                     ), num_nodes = self.rmol_n_node[j][idx])
              for j in range(self.rmol_max_cnt)]
              
        for j in range(self.rmol_max_cnt):
            g1[j].ndata['attr'] = torch.from_numpy(self.rmol_node_attr[j][self.rmol_n_csum[j][idx]:self.rmol_n_csum[j][idx+1]]).float()
            g1[j].edata['edge_attr'] = torch.from_numpy(self.rmol_edge_attr[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]]).float()
        
        g2 = [graph((self.pmol_src[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]],
                     self.pmol_dst[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]]
                     ), num_nodes = self.pmol_n_node[j][idx])
              for j in range(self.pmol_max_cnt)]

        for j in range(self.pmol_max_cnt):
            g2[j].ndata['attr'] = torch.from_numpy(self.pmol_node_attr[j][self.pmol_n_csum[j][idx]:self.pmol_n_csum[j][idx+1]]).float()
            g2[j].edata['edge_attr'] = torch.from_numpy(self.pmol_edge_attr[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]]).float()

        label = self.yld[idx]
        
        return *g1, *g2, label
        
        
    def __len__(self):

        return self.yld.shape[0]
