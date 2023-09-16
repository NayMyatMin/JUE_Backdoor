import os, torch, time, logging
from inversion import JUE_Backdoor
from tabulate import tabulate
from architecture.utils import MNIST_Network
from architecture.wrn import WideResNet
from preprocess.evaluate_data import Evaluate_Data

logging.basicConfig(format='%(message)s', level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

class Evaluate_Model:
    def __init__(self, model, submodel, model_file_path, true_target_label, attack_spec, args, dataloader) -> None:
        self.model = model
        self.submodel = submodel
        self.model_file_path = model_file_path
        self.true_target_label = true_target_label
        self.attack_spec = attack_spec
        self.args = args
        self.dataloader = dataloader
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
        return asr

    def evaluate(self, target):
        evaluate_data = Evaluate_Data(256, self.args.dataset, 'train', target, size_per_class=5)
        x_val, y_val = evaluate_data.load_and_preprocess_data(self.dataloader)
        pattern = self.generate_backdoor(x_val, y_val, target)

        del x_val, y_val
        size = torch.norm(pattern, p=1)
        x_val, y_val = evaluate_data.load_and_preprocess_data(self.dataloader)
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
