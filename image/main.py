import argparse
import os
import pandas

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging, logging.config
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import json
import time 
import sys
from model import DRO_Loss
import math
def get_logger(name, log_dir="./log_pos/"):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read
    
    Returns
    -------
    A logger object which writes to both file and stdout
        
    """
    config_dict = json.load(open( "./config/log_config.json"))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

import utils
from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def train(net, data_loader, train_optimizer, temperature, estimator, tau_plus, beta, model_loss):
    net.train()
    weights = []
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    with logging_redirect_tqdm():
        for pos_1, pos_2, target, index in train_bar:
            pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)
            weight = None
            
            loss, weight = model_loss(out_1, out_2, index)
            if weight is not None:
                weights.append(weight)

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size

            if np.isnan(total_loss):
                logger.info("Nan is encountered!")
                sys.exit(1)
        if len(weights) != 0:
            weights = torch.cat(weights, dim=0)
            logger.info('Train Epoch: [{}/{}] Loss: {:.4f}, w_min:{:.4}, w_max:{:.4}'.format(epoch, epochs, total_loss / total_num, weights.min().item(), weights.max().item()))
        else:
            logger.info('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        with logging_redirect_tqdm():
        # generate feature bank
            for data, _, target, _ in tqdm(memory_data_loader, desc='Feature extracting'):
                feature, out = net(data.to(device, non_blocking=True))
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            if 'cifar' in dataset_name:
                feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
            elif 'stl' in dataset_name:
                feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 

            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(test_data_loader)
            for data, _, target, _ in test_bar:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                feature, out = net(data)

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()

            logger.info('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                    .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
                # test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                #                         .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--name', default='BEST', type=str, help='Choose loss function')
    parser.add_argument('--root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='hard', type=str, help='Choose loss function')
    parser.add_argument('--dataset_name', default='stl10', type=str, help='Choose loss function')
    parser.add_argument('--beta', default=1.0, type=float, help='Choose loss function')
    parser.add_argument('--anneal', default=None, type=str, help='Beta annealing')
    parser.add_argument('--num_workers', default=12, type=int, help='num_workers')
    parser.add_argument('--seed', default=12, type=int, help='num_workers')


    # args parse
    args = parser.parse_args()
    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, estimator = args.batch_size, args.epochs,  args.estimator
    dataset_name = args.dataset_name
    beta = args.beta
    anneal = args.anneal
    print("set seed!!!\t{}".format(args.seed))
    set_all_seeds(args.seed)
    # torch.save(model.state_dict(), '../results/{}/{}_{}_model_T{}_{}_{}_{}_{}.pth'.format(dataset_name,dataset_name,estimator, args.temperature,batch_size,tau_plus,beta,epoch))
    # args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
    name = args.name + ".log"
    global logger
    logger = get_logger(name)
    logger.info(vars(args))
    #configuring an adaptive beta if using annealing method
    if anneal=='down':
        do_beta_anneal=True
        n_steps=9
        betas=iter(np.linspace(beta,0,n_steps))
    else:
        do_beta_anneal=False
    
    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)
    if args.dataset_name == 'cifar10': 
        data_size = 60000 + 1
    elif args.dataset_name == 'stl10': 
        data_size = 150000+1 
    else:
        data_size = 1000000 

    model_loss = DRO_Loss(temperature, tau_plus, batch_size, beta, estimator, N=data_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))
    print("T:{}".format(args.temperature))
    # training loop
    dataset_logdir='./results/{}'.format(dataset_name)
    if not os.path.exists(dataset_logdir):
        os.makedirs(dataset_logdir)

    for epoch in range(1, epochs + 1):
    # for epoch in range(1):
        train_loss = train(model, train_loader, optimizer, temperature, estimator, tau_plus, beta, model_loss)
        
        if do_beta_anneal is True:
            if epoch % (int(epochs/n_steps)) == 0:
                beta=next(betas)

        if epoch % 100 == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            if do_beta_anneal is True:
                torch.save(model.state_dict(), './results/{}/{}_{}_CNT_Epoch_{}.pth'.format(dataset_name, args.name, anneal, epoch))
            else:
                torch.save(model.state_dict(), './results/{}/{}_CNT_Epoch_{}.pth'.format(dataset_name, args.name, epoch))
