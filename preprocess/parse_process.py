import argparse, logging
import os, random, torch
import numpy as np
import torch.nn as nn

from architecture.utils import MNIST_Network
from architecture.wrn import WideResNet

from collections import OrderedDict
logging.basicConfig(format='%(message)s', level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Parse_Process: 
    def __init__(self) -> None:
        self.args = self.parse_and_set_arguments()
        self.model_path = self.set_model_path(self.args.dataset)
        self.phase = self.args.phase
        self.dataset = self.args.dataset
        self.batch_size = self.args.batch_size
        self.num_classes = self.args.num_classes
        self.attack_size = self.args.attack_size
        self.device = torch.device('cuda')

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Process input arguments.')
        parser.add_argument('--gpu',   default='0',        help='gpu id')
        parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST', help='dataset')
        parser.add_argument('--phase', choices=['evaluate'], default='evaluate', help='phase of framework')
        parser.add_argument('--seed',        default=1024, type=int, help='random seed')
        parser.add_argument('--batch_size',  default=32,   type=int, help='batch size')
        parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
        parser.add_argument('--attack_size', default=50,   type=int, help='number of samples for inversion')
        args = parser.parse_args()
        return args

    def parse_and_set_arguments(self):
        args = self.parse_arguments()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        return args
    
    def set_model_path(self, dataset):
        model_paths = {'MNIST': './dataset/M-Blended/', 'CIFAR10': './dataset/C-Patch-1/'}
        return model_paths.get(dataset, 'default_path/')
    
    def get_sub_dirs(self):
        sub_dirs = [os.path.join(self.model_path, sub_dir) for sub_dir in os.listdir(self.model_path)
                if os.path.isdir(os.path.join(self.model_path, sub_dir))]
        sub_dirs.sort()
        return sub_dirs
    
    def get_model_and_attack_spec(self, dir, files):
        model_file_path, attack_spec = None, None
        for file in files:
            if file == "model.pt":
                model_file_path = os.path.join(dir, file)
            if file == "attack_specification.pt":
                attack_spec = os.path.join(dir, file)
        return model_file_path, attack_spec

    def load_model(self, model_class, name, *args, **kwargs):
        if not os.path.exists(name):
            raise FileNotFoundError(f"No such model file: {name}")
        model = model_class(*args, **kwargs)
        loaded_model = torch.load(name)
        state_dict = loaded_model if isinstance(loaded_model, OrderedDict) else loaded_model.state_dict()
        model.load_state_dict(state_dict)
        return model

    def load_model_based_on_dataset(self, model_file_path):
        if self.args.dataset == 'MNIST':
            return self.load_model(MNIST_Network, model_file_path).to(self.device)
        elif self.args.dataset == 'CIFAR10':
            depth, num_classes, widen_factor, dropRate = 40, 10, 2, 0.0
            return self.load_model(WideResNet, model_file_path, depth, num_classes, widen_factor, dropRate).to(self.device)
        else:
            raise ValueError(f"Unsupported dataset type: {self.args.dataset}")

    def process_directory(self, dir, files):
        model_file_path, attack_spec_path = self.get_model_and_attack_spec(dir, files)
        if model_file_path and attack_spec_path:
            true_target_label = torch.load(attack_spec_path)["target_label"]
            logging.info(f'Model: {model_file_path.rsplit("/", 2)[1]}, True_Target: {true_target_label}')
            model = self.load_model_based_on_dataset(model_file_path)
            model.eval()
            logging.info('Load model from: {}'.format(self.model_path))
            return model, model_file_path, true_target_label
        
    def get_submodel(self, model):
        if isinstance(model, MNIST_Network):
            sub_model_layers = list(model.main.children())[:-1]
        elif isinstance(model, WideResNet):
            sub_model_layers = list(model.children())[:-1]
        else:
            raise ValueError("Unsupported model type.")
        sub_model = nn.Sequential(*sub_model_layers)
        sub_model.eval()
        return sub_model