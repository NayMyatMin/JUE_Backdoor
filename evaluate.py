import os, torch, time, logging
from inversion import JUE_Backdoor
from preprocess.dataset import MNISTData, CIFAR10Data

import numpy as np
from tabulate import tabulate
from collections import defaultdict
from architecture.utils import MNIST_Network
from architecture.wrn import WideResNet
logging.basicConfig(format='%(message)s', level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ResultLogger:
    def __init__(self, num_classes):
        self.l0_norm_list = torch.zeros(num_classes)
        self.all_results = []

    def log_single_target(self, target, size, asr):
        self.l0_norm_list[target] = size
        self.all_results.append({'Target': target, 'Trigger Size': f"{size:.2f}", 'Success Rate': f"{asr:.2f}"})

    def print_final_results(self):
        headers = ["Metrics"] + [str(result["Target"]) for result in self.all_results]
        trigger_sizes = ["Trigger Size"] + [result["Trigger Size"] for result in self.all_results]
        success_rates = ["Success Rate"] + [result["Success Rate"] for result in self.all_results]
        table_data = [trigger_sizes, success_rates]
        logging.info('\t')
        logging.info(tabulate(table_data, headers=headers, tablefmt='grid', numalign="right"))
        logging.info(self.l0_norm_list)

from collections import defaultdict
import torch
import numpy as np

def get_balanced_data(loader, exclude_label, size_per_class, num_batches=10):
    x_data = defaultdict(list)
    y_data = defaultdict(list)
    class_counters = defaultdict(int)

    # Accumulate data over multiple batches to ensure sufficient samples
    for _ in range(num_batches):
        x_batch, y_batch = loader.get_next_batch()
        for x, y in zip(x_batch, y_batch):
            if y != exclude_label:
                x_data[y.item()].append(x)  
                y_data[y.item()].append(y)
                class_counters[y.item()] += 1

    balanced_x_data = []
    balanced_y_data = []

    # Create balanced dataset based on the size_per_class
    for cls, samples in x_data.items():
        if class_counters[cls] >= size_per_class:
            balanced_x_data.extend(samples[:size_per_class])
            balanced_y_data.extend([cls] * size_per_class)

    # Handle the case where no balanced data could be formed
    if not balanced_x_data:
        print("Error: No balanced data could be created.")
        return None, None

    # Convert lists to PyTorch tensors
    balanced_x_data = torch.stack(balanced_x_data)
    balanced_y_data = torch.tensor(balanced_y_data, dtype=torch.long)

    # print(balanced_x_data.shape, balanced_y_data.shape)
    # class_counts = np.bincount(balanced_y_data)
    # for class_idx, count in enumerate(class_counts):
    #     print(f'Class {class_idx}: {count} samples')

    return balanced_x_data, balanced_y_data

def load_data(batch_size, dataset, mode, exclude_label=None):
    dataset_loader_map = {'MNIST': MNISTData, 'CIFAR10': CIFAR10Data}
    loader_class = dataset_loader_map.get(dataset)
    if exclude_label is not None:
        return loader_class(batch_size, mode, exclude_label)
    else:
        return loader_class(batch_size, mode)
    
def load_and_preprocess_data(dataset, batch_size, mode, exclude_label=None, size_per_class=10):
    data_loader = load_data(batch_size, dataset, mode)
    x_val, y_val = get_balanced_data(data_loader, exclude_label, size_per_class)
    y_val = torch.LongTensor(y_val)
    return x_val, y_val

def find_anchor_positions(model, target_class, k=1):
    if isinstance(model, MNIST_Network):
        last_layer_weights = model.main[-1].weight.data
    elif isinstance(model, WideResNet):
        last_layer_weights = model.fc.weight.data
    else:
        raise ValueError("Unsupported model type.")
    target_weights = last_layer_weights[target_class]
    _, sorted_indices = torch.sort(target_weights, descending=True)
    anchor_positions = sorted_indices[:k]
    return anchor_positions

class Evaluate_Model:
    def __init__(self, model, submodel, model_file_path, true_target_label, attack_spec, args) -> None:
        self.model = model
        self.submodel = submodel
        self.model_file_path = model_file_path
        self.true_target_label = true_target_label
        self.attack_spec = attack_spec
        self.args = args
        self.logger = ResultLogger(self.args.num_classes) 

    def generate_backdoor(self, x_val, y_val, target):
        shape = (1, 28, 28) if self.args.dataset == 'MNIST' else (3, 32, 32)
        anchor_positions = find_anchor_positions(self.model, target, k=1)
        backdoor = JUE_Backdoor(self.model, self.submodel, anchor_positions, shape, batch_size=self.args.batch_size)
        return backdoor.generate(target, x_val, y_val, attack_size=self.args.attack_size)
    
    def evaluate_attack(self, pattern, x_val, target):
        x_val = x_val.detach().to(self.args.device)        
        x_adv = torch.clamp(x_val + pattern, 0, 1)
        pred = self.model(x_adv).argmax(dim=1)        
        correct = (pred == target).sum().item()
        asr = correct / pred.size(0)
        
        del x_val, x_adv, pred
        torch.cuda.empty_cache()
        
        return asr

    def evaluate(self, target):
        x_val, y_val = load_and_preprocess_data(self.args.dataset, 512, 'train', target)
        pattern = self.generate_backdoor(x_val, y_val, target)

        del x_val, y_val
        torch.cuda.empty_cache()

        size = torch.norm(pattern, p=1)
        x_val, y_val = load_and_preprocess_data(self.args.dataset, 512, 'train', target)
        asr = self.evaluate_attack(pattern, x_val, target)
        return size, asr

    def evaluate_and_log_single_target(self, target):
        size, asr = self.evaluate(target)
        self.logger.log_single_target(target, size, asr)  

    def evaluate_all_targets(self):
        time_start = time.time()
        for target in range(self.args.num_classes):
            self.evaluate_and_log_single_target(target)
        self.logger.print_final_results()  
        logging.info(f'Generation Time: {(time.time() - time_start) / 60:.4f} m')
