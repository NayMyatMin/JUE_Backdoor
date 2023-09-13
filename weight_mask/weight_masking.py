import torch, time
import torch.nn.functional as F
from target_asr import TargetASR
from architecture.masked_conv import MaskedConv2d
from architecture.masked_mnist_net import Masked_MNIST_Network
from swm_utils import StateDictLoader, prepare_data, clip_mask, Regularization

def initialize_model(model_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_file_path, map_location=device)
    swm_model = Masked_MNIST_Network().to(device)
    StateDictLoader(swm_model, state_dict.state_dict()).load_state_dict()

    swm_model = swm_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for name, module in swm_model.named_modules():
        if isinstance(module, MaskedConv2d):
            module.include_mask()

    parameters = list(swm_model.named_parameters())
    mask_params = [v for n, v in parameters if "mask" in n]
    
    return swm_model, criterion, mask_params

def test(model, criterion, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)

    return loss, acc

def calculate_adversarial_perturbation(model, images, batch_pert, batch_opt):
    ori_lab = torch.argmax(model.forward(images), axis=1).long()
    per_logits = model.forward(images + batch_pert)
    loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
    loss_regu = torch.mean(-loss)

    batch_opt.zero_grad()
    loss_regu.backward(retain_graph=True)
    batch_opt.step()

    trigger_norm = 1000
    pert = batch_pert * min(1, trigger_norm / torch.sum(torch.abs(batch_pert)))
    pert = pert.detach()

    return pert

def mask_train(model, criterion, mask_opt, data_loader):
    model.eval()
    total_correct, total_loss, nb_samples = 0, 0.0, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_pert = torch.zeros([1,1,28,28], requires_grad=True, device=device)

    batch_opt = torch.optim.SGD(params=[batch_pert], lr=10)

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        pert = calculate_adversarial_perturbation(model, images, batch_pert, batch_opt)
        nb_samples += images.size(0)
        perturbed_images = torch.clamp(images + pert[0], min=0, max=1)
        
        mask_opt.zero_grad()
        output_noise = model(perturbed_images)
        output_clean = model(images)
        pred = torch.argmax(output_clean, axis=1).long()

        loss_rob = criterion(output_noise, labels)
        loss_nat = criterion(output_clean, labels)
        L1, L2 = Regularization(model)

        alpha = 0.9; gamma = 1e-8
        print("loss_noise | ", loss_rob.item(), " | loss_clean | ", loss_nat.item(), " | L1 | ", L1.item())

        loss = alpha * loss_nat + (1 - alpha) * loss_rob + gamma * L1

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()

        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def train_test_model(swm_model, criterion, clean_val_loader, clean_test_loader, mask_optimizer, lr_scheduler, outer, inner_iters):
    initial_mask_params = {name: param.clone().detach() for name, param in swm_model.named_parameters() if 'mask' in name}
    for i in range(outer):
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(model=swm_model, criterion=criterion, data_loader=clean_val_loader, mask_opt=mask_optimizer)
        cl_test_loss, cl_test_acc = test(model=swm_model, criterion=criterion, data_loader=clean_test_loader)
        
        print('EPOCHS {} | CleanLoss {:.4f} | CleanACC {:.4f}'.format((i + 1) * inner_iters, cl_test_loss, cl_test_acc))
        
        # Check mask parameters changed for debuging purposes
        for name, param in swm_model.named_parameters():
            if 'mask' in name:
                diff = torch.sum(torch.abs(param - initial_mask_params[name]))
                print(f"Mask Parameter {name} has changed by: {diff.item()}")

        initial_mask_params = {name: param.clone().detach() for name, param in swm_model.named_parameters() if 'mask' in name}

        lr_decay = True
        if lr_decay:
            lr_scheduler.step()
    
def weight_masking(model, model_file_path, true_target_label, attack_spec):
    outer = 10; inner = 5; batch_size = 128
    clean_val_loader, clean_test_loader, inner_iters = prepare_data(batch_size, inner)
    swm_model, criterion, mask_params = initialize_model(model_file_path)

    mask_optimizer = torch.optim.Adam(mask_params, lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=mask_optimizer, gamma=0.9)

    start = time.time()
    train_test_model(swm_model, criterion, clean_val_loader, clean_test_loader, mask_optimizer, lr_scheduler, outer, inner_iters)

    print('Running time: {:.4f} s'.format(time.time() - start))
    TargetASR('MNIST').target_asr(model, swm_model, true_target_label, attack_spec)

if __name__ == "__main__":
    weight_masking()
