import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import AlexNet
import util
import torch.nn.utils.prune as prune
from scipy.sparse import csr_matrix

from net.quantization import apply_weight_sharing

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Define which model to use
#model = AlexNet().to(device)
model = torch.load("/Users/wuguanzhi/Desktop/hw6_Network Compression/assignment/saves/initial_model.ptmodel")
util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs):
    model.train()
    # print(f'{model.state_dict().keys()=}')
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            # for name, p in model.named_parameters():
            #     #################################
            #     # TODO:
            #     #    zero-out all the gradients corresponding to the pruned weights
            #     #################################
            #     pass
            for l,p in enumerate(model.parameters()):
                # print(f'{p.grad.data=}')
                gradient_mask = (p!=0).data
                # print(f'{gradient_mask =}')
                p.grad=gradient_mask.float() * p.grad.data
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data, target = data.to(device).float(), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


# Initial training
print("--- Initial training ---")
#train(args.epochs)
#torch.save(model, f"saves/initial_model.ptmodel")
accuracy = test()
util.log(f"initial_accuracy {accuracy}", args.log)
util.print_nonzeros(model)

print(f'{model.state_dict().keys()=}')
# Pruning
model.prune_by_percentile()
apply_weight_sharing(model)
# model.prune_by_std(args.sensitivity)
# model.prune_by_std()
print("--- After pruning ---")
accuracy = test()
util.log(f"accuracy_after_pruning {accuracy}", args.log)
util.print_nonzeros(model)

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(args.epochs)
torch.save(model, f"saves/model_after_retraining.ptmodel")
print("--- After Retraining ---")
util.print_nonzeros(model)
accuracy = test()
util.log(f"accuracy_after_retraining {accuracy}", args.log)
