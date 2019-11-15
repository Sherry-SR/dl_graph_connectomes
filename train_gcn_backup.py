import importlib
import argparse
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.init as init

from models.gcn.model import GcnNet
from utils.visualize import VisdomLinePlotter
from utils.data_handler import ConnectomeSet

def weights_init(m):
    if isinstance(m, GCNConv):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)
    if isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

def train(loader):
    model.train()
    error_all = 0
    loss_all = 0
    count = 0

    for i, input in enumerate(loader):
        optimizer.zero_grad()
        target = input.y
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        label = torch.argmax(output, dim = 1)

        loss = loss_function(output, target)
        loss.backward()
        loss_all += loss.item() * input.num_graphs

        error_all += torch.sum(label != target).item()
        #error_all += loss.item() * input.num_graphs

        count += input.num_graphs

        optimizer.step()
    
    return loss_all / count, error_all / count

def evaludate(loader):
    model.eval()
    error_all = 0
    loss_all = 0
    count = 0

    with torch.no_grad():
        for i, input in enumerate(loader):
            target = input.y
            input = input.to(device)
            output = model(input)
            target = target.to(device)
            label = torch.argmax(output, dim = 1)

            loss = loss_function(output, target)
            loss_all += loss.item() * input.num_graphs

            error_all += torch.sum(label != target).item()
            #error_all += loss.item() * input.num_graphs

            count += input.num_graphs
    return loss_all / count, error_all / count


DATA_PATH = "../Data/PNC_Enriched_connectomes_n761/num_streamlines"
LABEL_PATH = "../Data/PNC_Enriched_connectomes_n761/GO1_LTN_dtiQApass_enrichedConnectomes_20190408.xlsx"
OUTPUT_PATH = "../Data/PNC_Enriched_connectomes_n761"
fname = "num_streamlines_Sex.pkl"

num_epochs = 1000
batch_size = 64
in_channels = 1
out_channels = 2
num_nodes = 80

dataset = ConnectomeSet(OUTPUT_PATH, fname, [DATA_PATH, LABEL_PATH], num_nodes = num_nodes, label_name='Sex', factorize=True)

dataset = dataset.shuffle()
train_dataset = dataset[:int(len(dataset)*0.7)]
val_dataset = dataset[int(len(dataset)*0.7):]

print("num of training:", len(train_dataset))
print("num of validation:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GcnNet(in_channels, out_channels, num_nodes).to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-5)
lr_scheduler = StepLR(optimizer, step_size = 500, gamma = 0.1)
loss_function = torch.nn.CrossEntropyLoss()
plotter = VisdomLinePlotter('gcn')

model.apply(weights_init)

for epoch in range(num_epochs):
    lr_scheduler.step()

    train_loss, train_error = train(train_loader)
    val_loss, val_error = evaludate(val_loader)

    plotter.plot('loss', 'train', 'loss', epoch, train_loss, xlabel='Epochs')
    plotter.plot('loss', 'val', 'loss', epoch, val_loss, xlabel='Epochs')
    plotter.plot('error', 'train', 'error', epoch, train_error, xlabel='Epochs')
    plotter.plot('error', 'val', 'error', epoch, val_error, xlabel='Epochs')