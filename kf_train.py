import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from custom_train_tools import CustomDataset, train_model, system_log, data_normalize

# parsing user input option
parser = argparse.ArgumentParser(description='Train Implementation')
parser.add_argument('--num_layers', nargs='+', type=int,
                    default=[2, 8, 3], help='num layers')
parser.add_argument('--num_nodes', nargs='+', type=int,
                    default=[500, 800, 1600], help='num nodes')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--number_of_epoch', type=int, default=1000, help='train epoch')
parser.add_argument('--number_of_features', type=str, default="100", help='number of input features')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--custom_lr_schedule', type=bool, default=False, help='using custom lr scheduler')
parser.add_argument('--noise', type=bool, default=True, help='using noise or not')
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
    num_layers = args.num_layers
    nodes = args.num_nodes
    num_of_features = args.number_of_features
    in_nodes = int(num_of_features)*5
    custom_lr_schedule = args.custom_lr_schedule
    idx = args.index
    noise = args.noise

    # setting saving name of logfile, image and model weight
    if noise:
        file_name = "_noised"
    else:
        file_name = ""
    model_char = "{}_{}_{}_{}_{}_{}_{}_".format(
        nodes[0], nodes[1], nodes[2], num_layers[0], num_layers[1], num_layers[2], idx) + num_of_features + file_name

    # setting log file path and system logger
    log_file = './res_log/' + model_char + '.txt'
    system_logger = system_log(log_file)

    if (os.path.exists("mean_noised_" + num_of_features + ".npy") and os.path.exists(
            "std_noised_" + num_of_features + ".npy")) is False:
        data_train = np.loadtxt("kf_train" + file_name + "_" + num_of_features + ".csv", delimiter=",", dtype=np.float32)
        data_val = np.loadtxt("kf_val" + file_name + "_" + num_of_features + ".csv", delimiter=",", dtype=np.float32)
        data_train, mean, std = data_normalize(data_train, int(num_of_features)*5)
        data_val, _, _ = data_normalize(data_val, int(num_of_features)*5)
        np.save("mean" + file_name + "_" + num_of_features + ".npy", mean)
        np.save("std" + file_name + "_" + num_of_features + ".npy", std)
    else:
        mean = np.load("mean" + file_name + "_" + num_of_features + ".npy").tolist()
        std = np.load("std" + file_name + "_" + num_of_features + ".npy").tolist()
        data_train = np.loadtxt("kf_train" + file_name + "_" + num_of_features + ".csv", delimiter=",", dtype=np.float32)
        data_val = np.loadtxt("kf_val" + file_name + "_" + num_of_features + ".csv", delimiter=",", dtype=np.float32)
        data_train, _, _ = data_normalize(data_train, int(num_of_features)*5)
        data_val, _, _ = data_normalize(data_val, int(num_of_features)*5)

    # load train, validation dataset
    train_dataset = CustomDataset(data_train, int(num_of_features)*5)
    train_loader = DataLoader(dataset=train_dataset, pin_memory=True,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=60, drop_last=True)
    val_dataset = CustomDataset(data_val, int(num_of_features)*5)
    val_loader = DataLoader(dataset=val_dataset, pin_memory=True,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=60, drop_last=True)
    print("Data load complete")

    # train model
    model = train_model(num_layers, nodes, in_nodes, total_epoch, lr, train_loader, val_loader, model_char,
                        system_logger, custom_lr_schedule=False)
