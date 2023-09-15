from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from collections import OrderedDict
import numpy as np
import torch, random

class StateDictLoader:
    def __init__(self, model, orig_state_dict):
        self.model = model
        self.orig_state_dict = orig_state_dict

    def standardize_state_dict(self):
        if 'state_dict' in self.orig_state_dict.keys():
            self.orig_state_dict = self.orig_state_dict['state_dict']
        return self

    def populate_new_state_dict(self):
        new_state_dict = OrderedDict()
        for k, v in self.model.state_dict().items():
            if k in self.orig_state_dict.keys():
                new_state_dict[k] = self.orig_state_dict[k]
            elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
                new_state_dict[k] = self.orig_state_dict[k[:-6]].clone().detach()
            else:
                new_state_dict[k] = v
        self.new_state_dict = new_state_dict
        return self

    def load_state_dict(self):
        self.standardize_state_dict().populate_new_state_dict()
        self.model.load_state_dict(self.new_state_dict)

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)

def Regularization(model):
    L1=0
    L2=0
    for name, param in model.named_parameters():
        if 'mask' in name:
            L1 += torch.sum(torch.abs(param))
            L2 += torch.norm(param, 2)
    return L1, L2

def prepare_data(dataset_name, batch_size, inner, samples=500, num_workers=4):
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == 'MNIST':
        trainset = MNIST(root='../data', train=True, download=True, transform=transform)
        testset = MNIST(root='../data', train=False, download=True, transform=transform)
    
    elif dataset_name == 'CIFAR10':
        trainset = CIFAR10(root='../data', train=True, download=True, transform=transform)
        testset = CIFAR10(root='../data', train=False, download=True, transform=transform)

    x_train, y_train = trainset.data, trainset.targets
    x_test, y_test = testset.data, testset.targets

    # Explicitly convert to float32
    if dataset_name == 'MNIST':
        x_train = x_train.unsqueeze(1).to(dtype=torch.float32)
        x_test = x_test.unsqueeze(1).to(dtype=torch.float32)
    elif dataset_name == 'CIFAR10':
        x_train = torch.tensor(x_train.transpose((0, 3, 1, 2)), dtype=torch.float32)/255
        x_test = torch.tensor(x_test.transpose((0, 3, 1, 2)), dtype=torch.float32)/255  

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create Validation and Test Sets
    rand_idx = random.sample(list(range(len(y_test))), int(samples))
    ot_idx = [i for i in range(len(y_test)) if i not in rand_idx]
    
    clean_val = TensorDataset(x_test[rand_idx], y_test[rand_idx])
    clean_test = TensorDataset(x_test[ot_idx], y_test[ot_idx])
    
    # Create DataLoader Objects
    inner_iters = int(len(clean_val) / batch_size) * inner
    random_sampler = RandomSampler(data_source=clean_val, replacement=True, num_samples=inner_iters * batch_size)
    
    clean_val_loader = DataLoader(clean_val, batch_size=batch_size, shuffle=False, sampler=random_sampler, num_workers=num_workers)
    clean_test_loader = DataLoader(clean_test, batch_size=batch_size, num_workers=num_workers)
    
    return clean_val_loader, clean_test_loader, inner_iters
