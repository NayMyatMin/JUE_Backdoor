import torch
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class TargetASR:
    def __init__(self, dataset_name, root_path='./data', batch_size=1000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_name = dataset_name
        self.root_path = root_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.clean_loader = self.get_data_loader()

    def apply_trigger(self, x, attack_spec):
        trigger = attack_spec['trigger']
        pattern, mask, alpha = trigger['pattern'], trigger['mask'], trigger['alpha']
        pattern = pattern.to(self.device)
        mask = mask.to(self.device)
        x = mask * (alpha * pattern + (1 - alpha) * x) + (1 - mask) * x
        return x

    def get_data_loader(self):
        if self.dataset_name == 'MNIST':
            return DataLoader(MNIST(root=self.root_path, train=False, download=True, transform=self.transform), batch_size=self.batch_size, shuffle=False)
        elif self.dataset_name == 'CIFAR10':
            return DataLoader(CIFAR10(root=self.root_path, train=False, download=True, transform=self.transform), batch_size=self.batch_size, shuffle=False)
        else:
            raise ValueError("Invalid dataset name")

    def create_backdoored_dataset(self, target_label, attack_specification):
        backdoored_dataset = [(self.apply_trigger(clean_image.to(self.device), attack_specification), target_label) for clean_image, _ in self.clean_loader.dataset]
        return DataLoader(backdoored_dataset, batch_size=self.clean_loader.batch_size, shuffle=False)

    def calculate_accuracy(self, model, dataloader, target_label=None):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, predicted = torch.max(model(images).data, 1)
                total += labels.size(0)
                
                if target_label is None:
                    correct += (predicted == labels).sum().item()
                else:
                    correct += (predicted == target_label).sum().item()
        return correct / total * 100

    def target_asr(self, model, swm_model, true_target_label, attack_spec):
        attack_specification = torch.load(attack_spec)
        backdoored_loader = self.create_backdoored_dataset(true_target_label, attack_specification)
        
        results = {}
        results['Org_Accuracy'] = self.calculate_accuracy(model, self.clean_loader)
        results['Org_ASR'] = self.calculate_accuracy(model, backdoored_loader, target_label=true_target_label)
        
        results['SWM_Accuracy'] = self.calculate_accuracy(swm_model, self.clean_loader)
        results['SWM_ASR'] = self.calculate_accuracy(swm_model, backdoored_loader, target_label=true_target_label)
        
        print(results)
