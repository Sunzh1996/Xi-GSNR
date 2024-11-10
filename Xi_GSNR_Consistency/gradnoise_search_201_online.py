import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import os
from scores import get_score_func
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from statistics import mean
import time
from utils import add_dropout, init_network

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../_dataset/cifar10/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../201_api/NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/ICML', type=str, help='folder to save results')
parser.add_argument('--save_string', default='gradsign', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--maxofn', default=4, type=int, help='score is the max of this many evaluations of the network') ##
parser.add_argument('--kernel', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='5', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', default=True, action='store_true')
parser.add_argument('--activations', action='store_true')
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int,
                    help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=10, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

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
                # logging.info(mod.weight.grad.data.size())
                # logging.info(mod.weight.data.size())
                if mod.weight.grad is not None:
                    grad_dict[name]=[mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name,mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape( -1).numpy())
    return grad_dict

def caculate_gradsq_gradvar(grad_dict):
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    grad_square_var = 0
    # grad_var_sum = 0
    for j, modname in enumerate(grad_dict.keys()):
        grad_var = np.var(grad_dict[modname], axis=0)
        # print(grad_var)
        grad_square = np.mean(grad_dict[modname], axis=0)**2
        # print(grad_square)
        if grad_var.any()==0:
            grad_square_var += 0
        else:
            # grad_square_var += np.sum(grad_square/grad_var)
            grad_square_var += np.sum(grad_square/(grad_var + 5e-08)) #B64N4=1e-08  
    return grad_square_var

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat,
                                 args)
os.makedirs(args.save_loc, exist_ok=True)

times = []
chosen = []
topscores = []
order_fn = np.nanargmax

cifar10_test_acc = []
cifar10_val_acc = []
cifar100_test_acc = []
cifar100_val_acc = []
imagenet16_test_acc = []
imagenet16_val_acc = []

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

# metric = np.load('./results/final_results/gradnoise_noiseratio_nasbench201_cifar10__none_0.05_1_True_64_8_1_b64n8e08.npy')
# metric = np.load('./results/final_results/gradnoise_noiseratio_nasbench201_cifar10__none_0.05_1_True_2_10_1_b2n10e08.npy')
# acc = np.load('./results/final_results/gradnoise_accs_nasbench201_cifar10_True.npy')

runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    start = time.time()
    indices = np.random.randint(0, len(searchspace), args.n_samples)
    scores = []

    npstate = np.random.get_state()
    ranstate = random.getstate()
    torchstate = torch.random.get_rng_state()
    for i, arch in enumerate(indices):
        grad_dict= {}
        s = []
        
        print("{}/{}".format(i, len(indices)))
        # try:
        uid = searchspace[arch]

        network = searchspace.get_network(uid)
        network.to(device)
        if args.dropout:
            add_dropout(network, args.sigma)
        if args.init != '':
            init_network(network, args.init)

        random.setstate(ranstate)
        np.random.set_state(npstate)
        torch.set_rng_state(torchstate)

        # data_iterator = iter(train_loader)
        # x, target = next(data_iterator)
        # x, target = x.to(device), target.to(device)

        ##batch数据 maxofn次
        for j in range(args.maxofn):
            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            x, target = x.to(device), target.to(device)
            logits, out = network(x)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            grad_dict= get_grad_all(network, grad_dict, j)
        
        grad_square_var = caculate_gradsq_gradvar(grad_dict)
        if grad_square_var==0:
            s.append(0)
        else:
            s.append(grad_square_var)


        scores.append(np.mean(s))

    best_arch = indices[order_fn(scores)]
    uid = searchspace[best_arch]
    topscores.append(scores[order_fn(scores)])
    chosen.append(best_arch)
    # # acc.append(searchspace.get_accuracy(uid, acc_type, args.trainval))
    # acc.append(searchspace.get_final_accuracy(uid, acc_type, False))

    # if not args.dataset == 'cifar10' or args.trainval:
    #     val_acc.append(searchspace.get_final_accuracy(uid, val_acc_type, args.trainval))
    # # val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

    cifar10_val_acc.append(searchspace.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', 'x-valid')['accuracy'])
    cifar10_test_acc.append(searchspace.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10', 'ori-test')['accuracy']) 

    cifar100_val_acc.append(searchspace.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar100', 'x-valid')['accuracy'])
    cifar100_test_acc.append(searchspace.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar100', 'x-test')['accuracy'])

    imagenet16_val_acc.append(searchspace.api.query_meta_info_by_index(uid, hp='200').get_metrics('ImageNet16-120', 'x-valid')['accuracy'])
    imagenet16_test_acc.append(searchspace.api.query_meta_info_by_index(uid, hp='200').get_metrics('ImageNet16-120', 'x-test')['accuracy'])

    times.append(time.time() - start)
    # runs.set_description(f"acc: {mean(acc):.2f}% time:{mean(times):.2f}")
    runs.set_description(f"cifar10_val_acc: {mean(cifar10_val_acc):.2f}% time:{mean(times):.2f}")
    runs.set_description(f"cifar10_test_acc: {mean(cifar10_test_acc):.2f}% time:{mean(times):.2f}")
    runs.set_description(f"cifar100_val_acc: {mean(cifar100_val_acc):.2f}% time:{mean(times):.2f}")
    runs.set_description(f"cifar100_test_acc: {mean(cifar100_test_acc):.2f}% time:{mean(times):.2f}")
    runs.set_description(f"imagenet16_val_acc: {mean(imagenet16_val_acc):.2f}% time:{mean(times):.2f}")
    runs.set_description(f"imagenet16_test_acc: {mean(imagenet16_test_acc):.2f}% time:{mean(times):.2f}")


# print(f"Final mean test accuracy: {np.mean(acc)}")
print(f"Final mean cifar10 validation accuracy: {np.mean(cifar10_val_acc)}")
print(f"Final mean cifar10 test accuracy: {np.mean(cifar10_test_acc)}")
print(f"Final mean cifar100 validation accuracy: {np.mean(cifar100_val_acc)}")
print(f"Final mean cifar100 test accuracy: {np.mean(cifar100_test_acc)}")
print(f"Final mean iamgenet16 validation accuracy: {np.mean(imagenet16_val_acc)}")
print(f"Final mean iamgenet16 test accuracy: {np.mean(imagenet16_test_acc)}")

# if len(val_acc) > 1:
#    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

# state = {'accs': acc,
#          'chosen': chosen,
#          'times': times,
#          'topscores': topscores,
#          }

# dset = args.dataset if not (args.trainval and args.dataset == 'cifar10') else 'cifar10-valid'
# fname = f"{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{dset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
# torch.save(state, fname)
