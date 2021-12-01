import numpy as np
import sys, csv, os
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from model import training, inference

from dataset import GraphDataset
from util import collate_reaction_graphs
from model import reactionMPNN

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats



data_id = 1 #data_id 1: Buchwald-Hartwig, #data_id 2: Suzuki-Miyaura, %data_id 3: out-of-sample test splits for Buchwald-Hartwig
split_id = 0 #data_id 1 & 2: 0-9, data_id 3: 1-4 
train_size = 2767 #data_id 1: [2767, 1977, 1186, 791, 395, 197, 98], data_id 2: [4032, 2880, 1728, 1152, 576, 288, 144], data_id 3: [3057, 3055, 3058, 3055]
batch_size = 128
use_saved = False
model_path = './model/model_%d_%d_%d.pt' %(data_id, split_id, train_size)

data = GraphDataset(data_id, split_id)
frac_split = (train_size + 1e-5)/len(data)
train_set, test_set = split_dataset(data, [frac_split, 1 - frac_split], shuffle=False)

train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([batch_size, len(train_set)])), shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

print('-- CONFIGURATIONS')
print('--- data_type:', data_id, split_id)
print('--- train/test: %d/%d' %(len(train_set), len(test_set)))
print('--- max no. reactants:', data.rmol_max_cnt)
print('--- max no. products:', data.pmol_max_cnt)
print('--- use_saved:', use_saved)
print('--- model_path:', model_path)


# training 
train_y = np.hstack([inst[-1] for inst in iter(train_loader.dataset)])
train_y_mean = np.mean(train_y)
train_y_std = np.std(train_y)

node_dim = data.rmol_node_attr[0].shape[1]
edge_dim = data.rmol_edge_attr[0].shape[1]
net = reactionMPNN(node_dim, edge_dim).cuda()

if use_saved == False:
    print('-- TRAINING')
    net = training(net, train_loader, None, train_y_mean, train_y_std)
    torch.save(net.state_dict(), model_path)
else:
    print('-- LOAD SAVED MODEL')
    net.load_state_dict(torch.load(model_path))


# inference
test_y = np.hstack([inst[-1] for inst in iter(test_loader.dataset)])

test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std)
test_y_pred = np.clip(test_y_pred, 0, 100)

result = [mean_absolute_error(test_y, test_y_pred),
          mean_squared_error(test_y, test_y_pred) ** 0.5,
          r2_score(test_y, test_y_pred),
          stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
          
print('-- RESULT')
print('--- test size: %d' %(len(test_y)))
print('--- MAE: %.3f, RMSE: %.3f, R2: %.3f, Spearman: %.3f' %(result[0], result[1], result[2], result[3]))
