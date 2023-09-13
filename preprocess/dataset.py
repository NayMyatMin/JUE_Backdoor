from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
import sys, os

class BaseData:
    def __init__(self, batch_size, data_name, dataset_class, data_size, data_dir='./data'):
        self.batch_size = batch_size
        self.data_name = data_name
        self.data_dir = data_dir
        self.dataset_class = dataset_class
        self.size = data_size

        self.data = self.get_dataflow()
        self.iter = iter(self.data)

    def data_augmentor(self):
        return Compose([ToTensor()])

    def get_dataflow(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        is_train = True if self.data_name == 'train' else False
        transform = self.data_augmentor()
        sys.stdout = open(os.devnull, 'w')
        ds = self.dataset_class(root=self.data_dir, train=is_train, download=True, transform=transform)
        sys.stdout = sys.__stdout__
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=is_train, num_workers=4)
        return loader

    def get_next_batch(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data)
            batch = next(self.iter)
        return batch[0], batch[1]  # Returns images and labels
    
class MNISTData(BaseData):
    def __init__(self, batch_size, data_name='val'):
        super().__init__(batch_size, data_name, MNIST, {'train': 60000, 'val': 10000})

class CIFAR10Data(BaseData):
    def __init__(self, batch_size, data_name='val'):
        super().__init__(batch_size, data_name, CIFAR10, {'train': 50000, 'val': 10000})
