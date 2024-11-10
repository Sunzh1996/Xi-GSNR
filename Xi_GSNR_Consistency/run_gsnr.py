import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scipy import stats
from utils import add_dropout, init_network
import torch.nn.functional as F
import torch.nn as nn
import logging, sys

parser = argparse.ArgumentParser(description='NAS With GSNR')
parser.add_argument('--data_loc', default='../_dataset/cifar10/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../201_api/NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='gradnoise', type=str, help='prefix of results file')
parser.add_argument('--score', default='noiseratio', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='1', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', default=True, action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--batch_numbers', default=8, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--random_xi', default=1e-08, type=float)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=10, type=int, help='#classes (nasbench101)')
parser.add_argument('--start', type=int, default=0, help='start index')
parser.add_argument('--end', type=int, default=0, help='end index')
parser.add_argument('--save', type=str, default='./logs/gsnr_test.log', help='experiment name')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
   format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_grad_all(model:torch.nn.Module, grad_dict:dict, step_iter=0):
    if step_iter==0:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    grad_dict[name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict


def get_grad_batch(model:torch.nn.Module, grad_dict:dict, inputs, targets, loss_fn=F.cross_entropy):
    N = inputs.shape[0]
    for i in range(N):
        model.zero_grad()
        logits, out  = model.forward(inputs[[i]])
        loss = loss_fn(logits, targets[[i]])
        loss.backward()
        if i==0:
            for name,mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    grad_dict[name]=[mod.weight.grad.data.cpu().reshape( -1).numpy()]
        else:
            for name,mod in model.named_modules():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict


def caculate_gradsq_gradvar(grad_dict):
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    grad_square_var = 0
    for j, modname in enumerate(grad_dict.keys()):
        grad_var = np.var(grad_dict[modname], axis=0)
        grad_square = np.mean(grad_dict[modname], axis=0)**2
        if grad_var.any()==0:
            grad_square_var += 0
        else:
            grad_square_var += np.sum(grad_square/(grad_var + args.random_xi)) #B64N8=1e-08
    return grad_square_var


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
savedataset = args.dataset
dataset = 'fake' if 'fake' in args.dataset else args.dataset
args.dataset = args.dataset.replace('fake', '')
# if args.dataset == 'cifar10':
#     args.dataset = args.dataset + '-valid'
searchspace = nasspace.get_search_space(args)
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)

filename = f'{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{savedataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.batch_numbers}_{args.seed}'
accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{savedataset}_{args.trainval}'

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'


scores = []
accs = []
args.end = len(searchspace) if args.end == 0 else args.end
for i, (uid, network) in enumerate(searchspace):
    if i < args.start:
        continue
    if i >= args.end:
        break 

    if args.dropout:
        add_dropout(network, args.sigma)
    if args.init != '':
        init_network(network, args.init)

    network = network.to(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    s = []
    grad_dict= {}
    
    
    if args.nasspace == 'nasbench201':
        for j in range(args.batch_numbers):
            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            x, target = x.to(device), target.to(device)
            logits, out = network(x)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            grad_dict= get_grad_all(network, grad_dict, j)

    elif args.nasspace == 'nasbench101':
        data_iterator = iter(train_loader)
        x, target = next(data_iterator)
        x, target = x.to(device), target.to(device)
        grad_dict= get_grad_batch(network, grad_dict, inputs=x, targets=target, loss_fn=F.cross_entropy)


    grad_square_var = caculate_gradsq_gradvar(grad_dict)
    if grad_square_var==0:
        s.append(0)
    else:
        s.append(grad_square_var)


    scores.append(np.mean(s))
    accs.append(searchspace.get_final_accuracy(uid, acc_type, args.trainval))
    tau, p = stats.kendalltau(accs, scores, nan_policy='omit')
    logging.info(f'{uid} kendalltau = {tau}')
    spearman = abs(stats.spearmanr(accs, scores, nan_policy='omit').correlation)
    logging.info(f'{uid} spearmanr = {spearman}')

