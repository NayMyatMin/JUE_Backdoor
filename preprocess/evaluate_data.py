from preprocess.dataset import MNISTData, CIFAR10Data
from collections import defaultdict
import torch, numpy

class Evaluate_Data:
    def __init__(self, batch_size, dataset, mode, exclude_label=None, size_per_class=10):
        self.batch_size = batch_size
        self.dataset = dataset
        self.mode = mode
        self.exclude_label = exclude_label
        self.size_per_class = size_per_class
        self.dataset_loader_map = {'MNIST': MNISTData, 'CIFAR10': CIFAR10Data}

    def log_balanced_data_info(self, balanced_x_data, balanced_y_data):
        print(balanced_x_data.shape, balanced_y_data.shape)
        class_counts = numpy.bincount(balanced_y_data.numpy())  
        for class_idx, count in enumerate(class_counts):
            print(f'Class {class_idx}: {count} samples')

    def get_balanced_data(self, loader, num_batches=10):
        x_data = defaultdict(list)
        y_data = defaultdict(list)
        class_counters = defaultdict(int)

        for _ in range(num_batches):
            x_batch, y_batch = loader.get_next_batch()
            for x, y in zip(x_batch, y_batch):
                if y != self.exclude_label:
                    x_data[y.item()].append(x)
                    y_data[y.item()].append(y)
                    class_counters[y.item()] += 1

        balanced_x_data = []
        balanced_y_data = []

        for cls, samples in x_data.items():
            if class_counters[cls] >= self.size_per_class:
                balanced_x_data.extend(samples[:self.size_per_class])
                balanced_y_data.extend([cls] * self.size_per_class)

        if not balanced_x_data:
            print("Error: No balanced data could be created.")
            return None, None

        balanced_x_data = torch.stack(balanced_x_data)
        balanced_y_data = torch.tensor(balanced_y_data, dtype=torch.long)

        return balanced_x_data, balanced_y_data

    def load_data(self):
        loader_class = self.dataset_loader_map.get(self.dataset)
        return loader_class(self.batch_size, self.mode)

    def load_and_preprocess_data(self):
        data_loader = self.load_data()
        x_val, y_val = self.get_balanced_data(data_loader)
        y_val = torch.LongTensor(y_val)
        return x_val, y_val