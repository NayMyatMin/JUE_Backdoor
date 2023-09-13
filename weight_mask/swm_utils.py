from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision.datasets import MNIST
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

def prepare_data(batch_size, inner):
    # Load MNIST dataset
    trainset = MNIST(root='../data', train=True, download=True, transform=None)
    testset = MNIST(root='../data', train=False, download=True, transform=None)

    # Data preprocessing
    x_train, y_train = trainset.data, trainset.targets
    x_test, y_test = testset.data, testset.targets
    x_train = x_train.unsqueeze(1).to(dtype=torch.float32)  
    x_test = x_test.unsqueeze(1).to(dtype=torch.float32)    
    y_train = y_train.to(dtype=torch.long)
    y_test = y_test.to(dtype=torch.long)

    # Create validation and test sets
    num_samples = 500
    rand_idx = random.sample(list(np.arange(y_test.shape[0])), int(num_samples))
    ot_idx = [i for i in range(y_test.shape[0]) if i not in rand_idx]
    clean_val = TensorDataset(x_test[rand_idx].clone().detach(), y_test[rand_idx].clone().detach())
    clean_test = TensorDataset(x_test[ot_idx].clone().detach(), y_test[ot_idx].clone().detach())

    # Create DataLoader objects
    inner_iters = int(len(clean_val)/batch_size)*inner
    random_sampler = RandomSampler(data_source=clean_val, replacement=True, num_samples=inner_iters * batch_size)
    clean_val_loader = DataLoader(clean_val, batch_size=batch_size, shuffle=False, sampler=random_sampler, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=batch_size, num_workers=0)

    return clean_val_loader, clean_test_loader, inner_iters

