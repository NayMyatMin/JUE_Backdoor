import torch
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ApplyTrigger:
    def __init__(self, device):
        self.device = device

    def apply_trigger(self, x, attack_spec):
        trigger = attack_spec['trigger']
        pattern, mask, alpha = trigger['pattern'], trigger['mask'], trigger['alpha']
        pattern = pattern.to(self.device)
        mask = mask.to(self.device)
        x = mask * (alpha * pattern + (1 - alpha) * x) + (1 - mask) * x
        return x

def create_backdoored_dataset(clean_dataloader, target_label, attack_specification, device, applier):
    backdoored_dataset = [(applier.apply_trigger(clean_image.to(device), attack_specification), target_label) for clean_image, _ in clean_dataloader.dataset]
    return DataLoader(backdoored_dataset, batch_size=clean_dataloader.batch_size, shuffle=False)

def calculate_accuracy(model, dataloader, target_label=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            
            if target_label is None:  # Calculate clean accuracy
                correct += (predicted == labels).sum().item()
            else:  # Calculate ASR
                correct += (predicted == target_label).sum().item()
    return correct / total * 100

def get_data_loader(dataset_name, root_path, transform, batch_size=1000):
    if dataset_name == 'MNIST':
        return DataLoader(MNIST(root=root_path, train=False, download=True, transform=transform), batch_size=batch_size, shuffle=False)
    elif dataset_name == 'CIFAR10':
        return DataLoader(CIFAR10(root=root_path, train=False, download=True, transform=transform), batch_size=batch_size, shuffle=False)

def target_asr(model, true_target_label, attack_spec, dataset, dataset_root='./data', batch_size=1000, swm_model=None):
    transform = transforms.Compose([transforms.ToTensor()])
    
    clean_loader = get_data_loader(dataset, dataset_root, transform, batch_size)
    attack_specification = torch.load(attack_spec)
    
    # Create backdoored dataset and DataLoader
    applier = ApplyTrigger(device)
    backdoored_loader = create_backdoored_dataset(clean_loader, true_target_label, attack_specification, device, applier)
    
    # Calculate Metrics for the Original Model
    results = {}
    results['Org_Accuracy'] = calculate_accuracy(model, clean_loader)
    results['Org_ASR'] = calculate_accuracy(model, backdoored_loader, target_label=true_target_label)
    
    if swm_model is not None:
        results['SWM_Accuracy'] = calculate_accuracy(swm_model, clean_loader)
        results['SWM_ASR'] = calculate_accuracy(swm_model, backdoored_loader, target_label=true_target_label)
    
    print(results)
