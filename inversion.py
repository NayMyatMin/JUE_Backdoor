import numpy as np
import sys, torch

class JUE_Backdoor:
    def __init__(self, model, submodel, anchor_positions, shape, num_classes=10, steps=3000,
                batch_size=32, asr_bound=0.9, init_alpha=1e-3, lr=0.1, clip_max=1.0):

        self.model = model
        self.submodel = submodel
        self.anchor_positions = anchor_positions
        self.input_shape = shape
        self.num_classes = num_classes
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound
        self.init_alpha = init_alpha
        self.lr = lr
        self.clip_max = clip_max

        self.device = torch.device('cuda')
        self.epsilon = 1e-7
        self.patience = 10
        self.alpha_multiplier_up   = 1.5
        self.alpha_multiplier_down = 1.5 ** 1.5
        self.pattern_shape = self.input_shape

    def initialize_parameters(self):
        # store best results
        pattern_best     = torch.zeros(self.pattern_shape).to(self.device)
        pattern_pos_best = torch.zeros(self.pattern_shape).to(self.device)
        pattern_neg_best = torch.zeros(self.pattern_shape).to(self.device)
        reg_best = float('inf')
        pixel_best  = float('inf')

        # hyper-parameters to dynamically adjust loss weight
        alpha = self.init_alpha
        alpha = self.init_alpha
        alpha_up_counter   = 0
        alpha_down_counter = 0

        # initialize patterns with random values
        for i in range(2):
            init_pattern = np.random.random(self.pattern_shape) * self.clip_max
            init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
            init_pattern = init_pattern / self.clip_max

            if i == 0:
                pattern_pos_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_pos_tensor.requires_grad = True
            else:
                pattern_neg_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_neg_tensor.requires_grad = True
        return pattern_best, pattern_pos_best, pattern_neg_best, reg_best,\
            pixel_best, alpha, alpha, alpha_up_counter, alpha_down_counter, pattern_pos_tensor, pattern_neg_tensor

    def input_for_attack(self, target, x_set, y_set, attack_size=100):
        # For all-to-one attack
        indices = np.where(y_set != target)[0]  

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]

        x_set = x_set[indices].clone().detach().to(self.device)  
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        # avoid having the number of inputs smaller than batch size
        if attack_size < self.batch_size:
            self.batch_size = attack_size

        return x_set, y_set

    def remove_small_pattern(self, pattern_pos, pattern_neg, threshold):
        # remove small pattern values
        pattern_pos_cur = pattern_pos.detach()
        pattern_neg_cur = pattern_neg.detach()
        pattern_pos_cur[(pattern_pos_cur < threshold) & (pattern_pos_cur > -threshold)] = 0
        pattern_neg_cur[(pattern_neg_cur < threshold) & (pattern_neg_cur > -threshold)] = 0
        pattern_cur = pattern_pos_cur + pattern_neg_cur
        return pattern_cur
 
    def calculate_average(self, loss_reg_list, loss_list, acc_list):
        avg_loss_reg = np.mean(loss_reg_list)
        avg_loss     = np.mean(loss_list)
        avg_acc      = np.mean(acc_list)
        return avg_loss_reg, avg_loss, avg_acc

    def update_best_pattern(self, pattern_best, avg_acc, avg_loss_reg, pixel_cur, reg_best, pixel_best, \
            pattern_pos, pattern_neg, pattern_pos_tensor, pattern_neg_tensor, pattern_pos_best, pattern_neg_best, threshold):
        if avg_acc >= self.asr_bound and avg_loss_reg < reg_best and pixel_cur < pixel_best:
            reg_best = avg_loss_reg
            pixel_best = pixel_cur

            pattern_pos_best = pattern_pos.detach()
            pattern_pos_best[pattern_pos_best < threshold] = 0
            init_pattern = pattern_pos_best / self.clip_max
            with torch.no_grad():
                pattern_pos_tensor.copy_(init_pattern)

            pattern_neg_best = pattern_neg.detach()
            pattern_neg_best[pattern_neg_best > -threshold] = 0
            init_pattern = - pattern_neg_best / self.clip_max
            with torch.no_grad():
                pattern_neg_tensor.copy_(init_pattern)

            pattern_best = pattern_pos_best + pattern_neg_best
        return pattern_best, reg_best, pixel_best, pattern_pos_best, pattern_neg_best

    def adjust_alpha(self, avg_acc, alpha, alpha_up_counter, alpha_down_counter):
        # helper variables for adjusting loss weight
        if avg_acc >= self.asr_bound:
            alpha_up_counter += 1
            alpha_down_counter = 0
        else:
            alpha_up_counter = 0
            alpha_down_counter += 1

        # adjust loss weight
        if alpha_up_counter >= self.patience:
            alpha_up_counter = 0
            if alpha == 0:
                alpha = self.init_alpha
            else:
                alpha *= self.alpha_multiplier_up
        elif alpha_down_counter >= self.patience:
            alpha_down_counter = 0
            alpha /= self.alpha_multiplier_down
        return alpha, alpha_up_counter, alpha_down_counter

    def log_and_monitor(self, target, step, avg_acc, avg_loss, avg_loss_reg, reg_best, pixel_best):
        sys.stdout.write('\rTarget:{}, step: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                            .format(target, step, avg_acc, avg_loss)\
                            + 'reg: {:.2f}, reg_best: {:.2f}, '\
                            .format(avg_loss_reg, reg_best)\
                            + 'size: {:.0f}  '.format(pixel_best))
        sys.stdout.flush()

    def train_backdoor(self, x_set, y_set, pattern_pos_tensor, pattern_neg_tensor, alpha, optimizer):
            loss_reg_list, loss_list, acc_list = [], [], []
            delta_0 = None

            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                # map pattern variables to the valid range
                pattern_pos = 0.5 * (torch.tanh(pattern_pos_tensor) + 1) * self.clip_max
                pattern_neg = -0.5 * (torch.tanh(pattern_neg_tensor) + 1) * self.clip_max

                delta = pattern_pos + pattern_neg 
                if delta_0 is None:
                    delta_0 = delta.clone().detach()  

                x_adv = torch.clamp(x_batch + delta, min=0.0, max=self.clip_max)
                x_adv_0 = torch.clamp(x_batch + delta_0, min=0.0, max=self.clip_max)
                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                # Neuron Maximization Term 
                activations_delta = self.submodel(x_adv)[:, self.anchor_positions]
                activations_delta_0 = self.submodel(x_adv_0)[:, self.anchor_positions]
                neuron_max_term = torch.norm(activations_delta - 10 * activations_delta_0, p=2)
                reg_pos  = torch.max(torch.tanh(pattern_pos_tensor / 10) / (2 - self.epsilon) + 0.5, axis=0)[0]
                reg_neg  = torch.max(torch.tanh(pattern_neg_tensor / 10) / (2 - self.epsilon) + 0.5, axis=0)[0]
                loss_reg = torch.sum(reg_pos) + torch.sum(reg_neg)

                loss = neuron_max_term + alpha * loss_reg
                loss.backward()
                optimizer.step()

                # record loss and accuracy
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)
            return pattern_pos, pattern_neg, loss_reg_list, loss_list, acc_list

    def generate(self, target, x_set, y_set, attack_size=100):
        pattern_best, pattern_pos_best, pattern_neg_best, reg_best, pixel_best, alpha, alpha, \
            alpha_up_counter, alpha_down_counter, pattern_pos_tensor, pattern_neg_tensor = self.initialize_parameters()
        # y_set in this case is the target_set
        x_set, y_set = self.input_for_attack(target, x_set, y_set, attack_size)

        optimizer = torch.optim.Adam([pattern_pos_tensor, pattern_neg_tensor], lr=self.lr, betas=(0.5, 0.9))

        self.model.eval()
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            pattern_pos, pattern_neg, loss_reg_list, loss_list, acc_list = self.train_backdoor(
                x_set, y_set, pattern_pos_tensor, pattern_neg_tensor, alpha, optimizer)

            avg_loss_reg, avg_loss, avg_acc = self.calculate_average(loss_reg_list, loss_list, acc_list)

            threshold = self.clip_max / 255.0
            pattern_cur = self.remove_small_pattern(pattern_pos, pattern_neg, threshold)
            # pixel_cur = np.count_nonzero(np.sum(np.abs(pattern_cur.cpu().numpy()), axis=0))
            pixel_cur = torch.norm(pattern_cur.cpu(), p=1)

            pattern_best, reg_best, pixel_best, pattern_pos_best, pattern_neg_best  = self.update_best_pattern(
                pattern_best, avg_acc, avg_loss_reg, pixel_cur, reg_best, pixel_best, pattern_pos, pattern_neg, 
                        pattern_pos_tensor, pattern_neg_tensor, pattern_pos_best, pattern_neg_best, threshold)

            alpha, alpha_up_counter, alpha_down_counter = self.adjust_alpha(avg_acc, alpha, alpha_up_counter, alpha_down_counter)

            if step % 10 == 0:
                self.log_and_monitor(target, step, avg_acc, avg_loss, avg_loss_reg, reg_best, pixel_best)
                    
        sys.stdout.write('\x1b[2K')
        sys.stdout.flush()

        return pattern_best
