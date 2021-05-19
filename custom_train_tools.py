import time
import torch
import sys
import logging
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_normalize(data, num_of_features):  # define data normalize function
    data_reshaped = data[:, :num_of_features].reshape(-1, int(num_of_features / 17), 17)
    mean = [np.mean(data_reshaped[:, :, i]) for i in range(17)]
    std = [np.std(data_reshaped[:, :, i]) for i in range(17)]
    for i in range(17):
        data_reshaped[:, :, i] = (data_reshaped[:, :, i] - mean[i]) / std[i]
    data[:, :num_of_features] = data_reshaped.reshape(-1, num_of_features)

    return data, mean, std


def system_log(log_file):  # define system logger

    system_logger = logging.getLogger()
    system_logger.setLevel(logging.INFO)

    out_put_file_handler = logging.FileHandler(log_file)  # handler for log text file
    stdout_handler = logging.StreamHandler(sys.stdout)  # handler for log printing

    system_logger.addHandler(out_put_file_handler)
    system_logger.addHandler(stdout_handler)

    return system_logger


class CustomDataset(Dataset):  # define custom dataset (x: theta, elev, azim, r, vc), (y: r, vc, los, daz, dlos)
    def __init__(self, xy, num_of_features):
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy[:, :num_of_features])
        self.y_data = torch.tensor(xy[:, num_of_features:-1])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class FcLayerBn(nn.Module):  # define fully connected layer with Leaky ReLU activation function(for residual network)
    def __init__(self, in_nodes, nodes):
        super(FcLayerBn, self).__init__()
        self.fc = nn.Linear(in_nodes, nodes)
        self.bn1 = nn.BatchNorm1d(nodes)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.fc(x)
        out = self.bn1(out)
        out = self.act(out)
        return out


class WaveResNET(nn.Module):  # define custom model named wave net, which was coined after seeing the nodes sway(use residual to train)
    def __init__(self, block, planes, nodes, out_nodes=5, in_nodes=500):
        super(WaveResNET, self).__init__()
        self.in_nodes = in_nodes

        self.down_sample1 = self.down_sample(nodes[0])
        self.layer1 = self.make_layer(block, planes[0], nodes[0])
        self.down_sample2 = self.down_sample(nodes[1])
        self.layer2 = self.make_layer(block, planes[1], nodes[1])
        self.down_sample3 = self.down_sample(nodes[2])
        self.layer3 = self.make_layer(block, planes[2], nodes[2])

        self.fin_fc = nn.Linear(self.in_nodes, out_nodes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def make_layer(self, block, planes, nodes):

        layers = [block(self.in_nodes, nodes)]
        self.in_nodes = nodes
        for _ in range(1, planes):
            layers.append(block(self.in_nodes, nodes))

        return nn.Sequential(*layers)

    def down_sample(self, nodes):
        return nn.Sequential(nn.Linear(self.in_nodes, nodes), nn.BatchNorm1d(nodes))

    def forward_impl(self, x):
        identity = self.down_sample1(x)
        x = self.layer1(x)
        x = x.clone() + identity
        identity = self.down_sample2(x)
        x = self.layer2(x)
        x = x.clone() + identity
        identity = self.down_sample3(x)
        x = self.layer3(x)
        x = x.clone() + identity
        x = self.fin_fc(x)

        return x

    def forward(self, x):
        return self.forward_impl(x)


def train_model(num_layers, nodes, in_nodes, total_epoch, lr, train_loader, val_loader, model_char,
                system_logger, custom_lr_schedule=True):  # Function for train model

    model = WaveResNET(FcLayerBn, num_layers, nodes, in_nodes=in_nodes).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    if custom_lr_schedule:
        t_0 = 10
        t_mul = 2
        lr_gamma = 0.8
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, t_0, eta_min=0, last_epoch=-1)
        sch_step = 0
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    saving_path = "./res_model/"
    trn_loss_list = []  # list for saving train loss
    val_loss_list = []  # list for saving validation loss

    for epoch in range(total_epoch):
        trn_loss = 0.0

        # train model
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # grad init
            optimizer.zero_grad()
            # forward propagation
            output = model(inputs)
            # calculate loss
            loss = criterion(output, labels)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()
            # train loss summary
            trn_loss += loss.item()

        # validation
        with torch.no_grad():
            val_loss = 0.0
            for j, val in enumerate(val_loader):
                val_x, val_label = val
                val_x = val_x.to(device)
                val_label = val_label.to(device)
                val_output = model(val_x)
                v_loss = criterion(val_output, val_label)
                val_loss += v_loss
        # update lr scheduler
        if custom_lr_schedule:
            if sch_step == t_0:
                sch_step = 0
                t_0 *= t_mul
                optimizer.param_groups[0]['initial_lr'] = lr*lr_gamma
                lr = lr*lr_gamma
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, t_0, eta_min=0, last_epoch=-1)
            else:
                scheduler.step()
                sch_step += 1
        else:
            scheduler.step()

        # calculate average of train loss and validation loss for all batch
        trn_loss_list.append(trn_loss / len(train_loader))
        val_loss_list.append(val_loss / len(val_loader))
        now = time.localtime()

        # print logs
        system_logger.info("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year,
                                                              now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min,
                                                              now.tm_sec))
        system_logger.info("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} \n".format(
            epoch + 1, total_epoch, trn_loss /
            len(train_loader), val_loss / len(val_loader)))
    # save model
    model_name = saving_path + "Custom_model_" + model_char + "_fin.pth"
    torch.save(model, model_name)
    return model
