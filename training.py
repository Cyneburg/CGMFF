import random
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from model.Transformer import GATNet
from sklearn.metrics import roc_auc_score
from utils_test import *
from torch_geometric.data import DataLoader
from math import sqrt
from pytorchtools import EarlyStopping
from tqdm import trange

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(model, device, data_train, optimizer, epoch):
    print('\n Training on {} samples...'.format(len(data_train.dataset)))
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(data_train):
        data_mol = data[0].to(device)
        data_clique = data[1].to(device)
        y = data[0].y.view(-1, 1).to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_clique, None)
        loss = loss_fn(output, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} === {:.0f}% === Loss: {:.6f}'.format(epoch,
                                                                        100. * batch_idx / len(data_train),
                                                                        loss.item()))
    return total_loss

def predicting(model, device, data_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(data_test.dataset)))
    with torch.no_grad():
        for data in data_test:
            data_mol = data[0].to(device)
            data_clique = data[1].to(device)
            output = model(data_mol, data_clique, None)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data[0].y.view(-1, 1).cpu()), 0)
        return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2



TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000
Mode = "scaffold"

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

set = ['bace', 'bbbp']
for datafile in set:
    print(datafile)
    total_auc = []
    seed_everything()
    for random_seed in [42]:
        modeling = GATNet
        model = modeling(n_output=2, encoder=None).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        early_stopping = EarlyStopping(patience=30, verbose=True)

        data, smiles_list = creat_data(datafile,'train',random_seed)
        if Mode == 'scaffold':
            train_data, valid_data, test_data = scaffold_split(data, smiles_list, null_value=0, frac_train=0.8,
                                                               frac_valid=0.1, frac_test=0.1)
        elif Mode == 'random_scaffold':
            train_data, valid_data, test_data = random_scaffold_split(data, smiles_list, null_value=0, frac_train=0.8,
                                                                      frac_valid=0.1, frac_test=0.1, seed=random_seed)
        else:
            train_data, valid_data, test_data = random_split(data, smiles_list, null_value=0, frac_train=0.8,
                                                             frac_valid=0.1, frac_test=0.1, seed=random_seed)
        print('model', modeling)
        train_data = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE)
        valid_data = DataLoader(valid_data, batch_size=TRAIN_BATCH_SIZE)
        model_file_name = 'result/model/'+ datafile +'/model_{}_{}'.format(random_seed,Mode) +  '.model'

        best_AUC = 0
        for epoch in trange(NUM_EPOCHS):
            total_loss = train(model, device, train_data, optimizer, epoch + 1)
            T, S, Y = predicting(model, device, valid_data)
            AUC = roc_auc_score(T,S)
            if best_AUC < AUC:
                best_AUC = AUC
                torch.save(model.state_dict(), model_file_name)
            print('best AUC:{:.3f}'.format(best_AUC))
            early_stopping(total_loss)
            if early_stopping.early_stop:
                print('Early Stopping, The best AUC:{:.3f}'.format(best_AUC))
                break

        total_auc.append(best_AUC)
        with open('result/' + datafile + '/{}/result.txt'.format(Mode), 'a') as f:
            f.write(str(best_AUC) + '\n')

    avg_auc = sum(total_auc) / len(total_auc)
    print('Average AUC: {:.3f}'.format(avg_auc))


