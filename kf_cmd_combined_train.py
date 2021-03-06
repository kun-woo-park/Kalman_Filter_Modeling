import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from custom_train_tools_for_cmd_kf_combined import CustomDataset, train_model, system_log, data_normalize, FcLayerBn, WaveNET, WaveResNET
from custom_train_tools_for_cmd_kf_combined import FcLayer as FClayer

# fixing torch random seed
torch.manual_seed(0)

# parsing user input option
parser = argparse.ArgumentParser(description='Train Implementation')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--number_of_epoch', type=int, default=300, help='train epoch')
parser.add_argument('--number_of_features', type=str, default="100", help='number of input features')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--custom_lr_schedule', type=lambda s: s.lower() in ["true", 1], default=False, help='using custom lr scheduler')
parser.add_argument('--index', type=int, default=0, help='index(gpu number)')
args = parser.parse_args()

# designate GPU device number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.index)

# main loop
if __name__ == "__main__":
    # setting batch size and learning rate and total epoch
    batch_size = args.batch_size
    lr = args.lr
    total_epoch = args.number_of_epoch

    # setting number of layers and nodes by user input
    num_of_features = args.number_of_features
    in_nodes = int(num_of_features)*5
    custom_lr_schedule = args.custom_lr_schedule
    idx = args.index

    # setting load Kalman Filter model name
    model_name = "Custom_model_kf_" + num_of_features + ".pth"

    # setting saving name of logfile, image and model weight
    file_name = "_noised"
    model_char = "Combined" + file_name + "_" + num_of_features

    # setting log file path and system logger
    log_file = './res_log/' + model_char + '.txt'
    system_logger = system_log(log_file)

    # load data
    mean = np.load("mean" + file_name + "_" + num_of_features + ".npy").tolist()
    std = np.load("std" + file_name + "_" + num_of_features + ".npy").tolist()
    data_train = np.loadtxt("kf_train" + file_name + "_" + num_of_features + ".csv", delimiter=",", dtype=np.float32)
    data_val = np.loadtxt("kf_val" + file_name + "_" + num_of_features + ".csv", delimiter=",", dtype=np.float32)
    data_train, _, _ = data_normalize(data_train, int(num_of_features)*5)
    data_val, _, _ = data_normalize(data_val, int(num_of_features)*5)

    # load train, validation dataset
    train_dataset = CustomDataset(data_train, in_nodes)
    train_loader = DataLoader(dataset=train_dataset, pin_memory=True,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=60, drop_last=True)
    val_dataset = CustomDataset(data_val, in_nodes)
    val_loader = DataLoader(dataset=val_dataset, pin_memory=True,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=60, drop_last=True)
    print("Data load complete")

    # train model
    model = train_model(total_epoch, lr, train_loader, val_loader, model_char,
                        system_logger, model_name, custom_lr_schedule=True)  # Function for train model
