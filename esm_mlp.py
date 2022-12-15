import sys
sys.stdout = open('/home/log/epitope_classification.txt', 'w')
sys.stderr = open('/home/log/epitope_classification_err.txt', 'w')

import random
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings(action='ignore')

torch.cuda.empty_cache()

torch.cuda.is_available()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'NUM_WORKERS':4,
    'ANTIGEN_WINDOW':128,
    'ANTIGEN_MAX_LEN':128, # ANTIGEN_WINDOW와 ANTIGEN_MAX_LEN은 같아야합니다.
    'EPITOPE_MAX_LEN':256,
    'WINDOW':511, # Window - 1 for cls token
    'EPOCHS':10,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':2,
    'THRESHOLD':0.5,
    'DROPOUT':0.1,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

def get_preprocessing(data_type, new_df):
    # alpha_map = {
    #             '<CLS>':0, 
    #             '<PAD>':1, '<EOS>':2, '<unk>':3, 'L':4, 'A':5, 
    #             'G':6, 'V':7, 'S':8, 'E':9,'R':10, 
    #             'T':11, 'I':12, 'D':13, 'P':14, 'K':15, 
    #             'Q':16, 'N':17, 'F':18, 'Y':19, 'M':20,
    #             'H':21, 'W':22, 'C':23, 'X':24, 'B':25,
    #             'U':26, 'Z':27, 'O':28, '.':29, '-':30,
    #             '<null_1>':31, '<mask>':32
    #         }
    
    epitope_neighborhood_list = []

    for epitope, antigen, s_p, e_p in tqdm(zip(new_df['epitope_seq'], new_df['antigen_seq'], new_df['start_position'], new_df['end_position'])):
        epitope_neighborhood = ''
        
        epitope_len = len(epitope)
        s_p = int(s_p)
        e_p = int(e_p)
        
        start_p = s_p - int(CFG['WINDOW']/2 - (epitope_len//2))-1
        end_p = e_p + int(CFG['WINDOW']/2 - (epitope_len//2 + epitope_len%2) + CFG['WINDOW']%2 -1)
        
        if start_p < 0:
            start_p = 0
        if end_p > len(antigen):
            end_p = len(antigen)
        
        epitope_neighborhood += antigen[start_p:s_p -1]
        epitope_neighborhood += epitope
        epitope_neighborhood += antigen[e_p:end_p]

        epitope_neighborhood_list.append(epitope_neighborhood)
        
    label_list = None
    if data_type != 'test':
        label_list = []
        for label in new_df['label']:
            label_list.append(label)
    print(f'{data_type} dataframe preprocessing was done.')
    return epitope_neighborhood_list, label_list

all_df = pd.read_csv('../data/train.csv')
# Split Train : Validation = 0.8 : 0.2
all_df = all_df.sample(frac=1).reset_index(drop=True)
train_len = int(len(all_df)*0.8)
train_df = all_df.iloc[:train_len]
val_df = all_df.iloc[train_len:]

train_epitope_neighborhood_list, train_label_list = get_preprocessing('train', train_df)
val_epitope_neighborhood_list, val_label_list = get_preprocessing('val', val_df)

def make_zip(epitope_neighborhood_list, label_list):
    zips = []
    for epitope_neighborhood, label in zip(epitope_neighborhood_list, label_list):
        zips.append((label, epitope_neighborhood))
    return zips

train_data = make_zip(train_epitope_neighborhood_list, train_label_list)
valid_data = make_zip(val_epitope_neighborhood_list, val_label_list)

train_data = make_zip(train_epitope_neighborhood_list, train_label_list)
valid_data = make_zip(val_epitope_neighborhood_list, val_label_list)

import torch

encoder, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")

batch_converter = alphabet.get_batch_converter()

train_batch_labels, _ , train_batch_tokens = batch_converter(train_data)
valid_batch_labels, _ , valid_batch_tokens = batch_converter(valid_data)

print(train_batch_tokens.shape)

class Epitope_Dataset(Dataset):
    def __init__(self, tokens, labels):
        self.x = tokens
        self.y = labels
        self.len = len(self.x)
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.int32)
        y = torch.tensor(self.y[index], dtype=torch.float32)
        
        return x, y
    
train_dataset = Epitope_Dataset(tokens=train_batch_tokens, labels=train_batch_labels)
valid_dataset = Epitope_Dataset(tokens=valid_batch_tokens, labels=valid_batch_labels)

train_loader = DataLoader(train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True)

class ESM_MLP(nn.Module):
    def __init__(self, encoder, d_model, d_hidden, dropout):
        super(ESM_MLP, self).__init__()
        
        self.encoder = encoder
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        # self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=32, kernel_size=7, padding='same')
        # self.conv2 = nn.Conv1d(in_channels=self.d_model, out_channels=64, kernel_size=7, padding='same')
        # self.conv3 = nn.Conv1d(in_channels=self.d_model, out_channels=128, kernel_size=7, padding='same')
        
        # self.maxpoo1d = nn.MaxPool1d(7)
        
        self.fc1 = nn.Linear(self.d_model, self.d_hidden)
        self.gelu = nn.GELU()
        self.norm = nn.BatchNorm1d(self.d_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(self.d_hidden, 1)
        
    def forward(self, inputs):
        
        output = self.encoder(inputs, repr_layers=[33], return_contacts=False)
        
        # print(output["representations"][33][:, 0])
        # print(output["representations"][33][:, 0].shape)
        
        output = self.fc1(output["representations"][33][:, 0])
        output = self.gelu(output)
        output = self.norm(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = F.sigmoid(output)
        
        return output
    
model = ESM_MLP(encoder=encoder, d_model=1280, d_hidden=512, dropout=CFG["DROPOUT"])
model = nn.DataParallel(model)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=1e-5)

criterion = nn.BCELoss().to(DEVICE)

def train(model, train_loader, optimizer, criterion, log_interval, epoch, batch_size, DEVICE):
    model.train()
    
    train_loss = []
    
    preds = []
    targets = []
    
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        inputs = x.to(DEVICE)
        target = y.reshape(-1, 1).to(DEVICE)
        
        y_pred = model(inputs)
        loss = criterion(y_pred, target)
        train_loss.append(loss.tolist())
        
        loss.backward()
        optimizer.step()
        
        preds += torch.where(y_pred > CFG['THRESHOLD'], 1, 0).tolist()
        targets += target.to('cpu').tolist()
        
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\t Loss: {}\t F1 Score: {:.6f}\t".format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader), np.mean(train_loss), f1_score(targets, preds, average='macro')), end = "\r", flush=True)
            
def evaluate(model, test_loader, criterion, DEVICE, batch_size):
    model.eval()
    
    test_loss = []
    preds = []
    targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
        
            inputs = x.to(DEVICE)
            target = y.reshape(-1,1).to(DEVICE)

            y_pred = model(inputs)
            test_loss.append(criterion(y_pred, target).tolist())
            
            preds += torch.where(y_pred > CFG['THRESHOLD'], 1, 0).tolist()
            targets += target.to('cpu').tolist()
    
    return np.mean(test_loss),  f1_score(targets, preds, average='macro')

best = 0
for epoch in range(1, CFG['EPOCHS'] + 1):
    train(model=model, train_loader=train_loader, optimizer=optimizer, 
              log_interval=10, epoch=epoch, batch_size=CFG['BATCH_SIZE'], criterion=criterion, DEVICE=DEVICE)    
    valid_loss, valid_f1 = evaluate(model, test_loader=valid_loader, criterion=criterion, DEVICE=DEVICE, batch_size=CFG['BATCH_SIZE'])
    if valid_f1 > best:
        best = valid_f1
        torch.save(model, "../models/esm_best_loss.pt") 
        
    print("\n[EPOCH: {}], \tValid Loss: {: .6f}\tValid F1 Score: {:.6f}\n".format(epoch, valid_loss, valid_f1))