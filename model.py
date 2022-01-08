import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import dgl
from dgl.nn.pytorch import NNConv, Set2Set

from util import MC_dropout
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats


class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 readout_feats = 1024,
                 predict_hidden_feats = 512, prob_dropout = 0.1):
        
        super(reactionMPNN, self).__init__()

        self.mpnn = MPNN(node_in_feats, edge_in_feats)

        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 2, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 2)
        )
    
    def forward(self, rmols, pmols):

        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)

        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out = self.predict(concat_feats)

        return out[:,0], out[:,1]

        
def training(net, train_loader, val_loader, train_y_mean, train_y_std, val_monitor_epoch = 10, n_forward_pass = 5, cuda = torch.device('cuda:0')):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    
    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt
    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = nn.MSELoss(reduction = 'none')

    n_epochs = 500
    optimizer = Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-5)
    lr_scheduler = MultiStepLR(optimizer, milestones = [400, 450], gamma = 0.1, verbose = False)

    for epoch in range(n_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+pmol_max_cnt]]
            
            labels = (batchdata[-1] - train_y_mean) / train_y_std
            labels = labels.to(cuda)
            
            pred, logvar = net(inputs_rmol, inputs_pmol)

            loss = loss_fn(pred, labels)
            loss = (1 - 0.1) * loss.mean() + 0.1 * ( loss * torch.exp(-logvar) + logvar ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item()

        print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
              %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, train_loss, (time.time()-start_time)/60))
              
        lr_scheduler.step()

        # validation
        if val_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            
            val_y = val_loader.dataset.dataset.y[val_loader.dataset.indices]
            val_y_pred, _, _ = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)

            result = [mean_absolute_error(val_y, val_y_pred),
                      mean_squared_error(val_y, val_y_pred) ** 0.5,
                      r2_score(val_y, val_y_pred)]
                      
            print('--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(val_y), result[0], result[1], result[2]))

    print('training terminated at epoch %d' %epoch)
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cuda:0')):

    batch_size = test_loader.batch_size
    
    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt
    except:
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.pmol_max_cnt
             
    net.eval()
    MC_dropout(net)
    
    test_y_mean = []
    test_y_var = []
    
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+pmol_max_cnt]]

            mean_list = []
            var_list = []
            
            for _ in range(n_forward_pass):
                mean, logvar = net(inputs_rmol, inputs_pmol)
                mean_list.append(mean.cpu().numpy())
                var_list.append(np.exp(logvar.cpu().numpy()))

            test_y_mean.append(np.array(mean_list).transpose())
            test_y_var.append(np.array(var_list).transpose())

    test_y_mean = np.vstack(test_y_mean) * train_y_std + train_y_mean
    test_y_var = np.vstack(test_y_var) * train_y_std ** 2
    
    test_y_pred = np.mean(test_y_mean, 1)
    test_y_epistemic = np.var(test_y_mean, 1)
    test_y_aleatoric = np.mean(test_y_var, 1)
    
    return test_y_pred, test_y_epistemic, test_y_aleatoric
