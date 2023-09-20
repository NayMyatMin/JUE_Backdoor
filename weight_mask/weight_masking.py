import torch, time
import torch.nn.functional as F
from target_asr import TargetASR
from architecture.masked_conv import MaskedConv2d
from architecture.masked_wrn import Masked_WideResNet
from architecture.masked_mnist_net import Masked_MNIST_Network
from swm_utils import StateDictLoader, prepare_data, clip_mask, Regularization

def initialize_model(model_file_path, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_file_path, map_location=device)
    
    if dataset_name == 'MNIST':
        swm_model = Masked_MNIST_Network().to(device)
    elif dataset_name == 'CIFAR10':
        depth, num_classes, widen_factor, dropRate = 40, 10, 2, 0.0
        swm_model = Masked_WideResNet(depth, num_classes, widen_factor, dropRate).to(device)

    StateDictLoader(swm_model, state_dict.state_dict()).load_state_dict()

    swm_model = swm_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for name, module in swm_model.named_modules():
        if isinstance(module, MaskedConv2d):
            module.include_mask()

    # Freeze all parameters
    for param in swm_model.parameters():
        param.requires_grad = False

    parameters = list(swm_model.named_parameters())
    mask_params = [v for n, v in parameters if "mask" in n]

    # Unfreeze only the mask parameters
    for param in mask_params:
        param.requires_grad = True
    
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

def mask_train(model, criterion, mask_opt, data_loader, trigger):
    model.train()
    total_correct, total_loss, nb_samples = 0, 0.0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trigger = trigger.to(device)
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)
        perturbed_images = torch.clamp(images + trigger, min=0, max=1)
        
        mask_opt.zero_grad()
        output_noise = model(perturbed_images)
        output_clean = model(images)
        pred = torch.argmax(output_clean, axis=1).long()

        loss_rob = criterion(output_noise, labels)
        loss_nat = criterion(output_clean, labels)
        L1, L2 = Regularization(model)  

        eta = 0.9; zeta = 1 - eta; lambda_reg = 1e-8

        if i % 300 == 0:
            print("loss_noise | ", loss_rob.item(), " | loss_clean | ", loss_nat.item(), " | L1 | ", L1.item())

        loss = zeta * loss_nat + eta * loss_rob + lambda_reg * L1

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()

        mask_opt.step()
        clip_mask(model)  

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def train_test_model(swm_model, criterion, clean_val_loader, clean_test_loader, mask_optimizer, lr_scheduler, outer, inner_iters, trigger):
    initial_mask_params = {name: param.clone().detach() for name, param in swm_model.named_parameters() if 'mask' in name}
    for i in range(outer):
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(model=swm_model, criterion=criterion, data_loader=clean_val_loader, mask_opt=mask_optimizer, trigger=trigger)
        cl_test_loss, cl_test_acc = test(model=swm_model, criterion=criterion, data_loader=clean_test_loader)
        print('EPOCHS {} | CleanLoss {:.4f} | CleanACC {:.4f}'.format((i + 1) * inner_iters, cl_test_loss, cl_test_acc))
        
        # Check mask parameters changed for debuging purposes
        for name, param in swm_model.named_parameters():
            if 'mask' in name:
                diff = torch.sum(torch.abs(param - initial_mask_params[name]))
                # print(f"Mask Parameter {name} has changed by: {diff.item()}")

        initial_mask_params = {name: param.clone().detach() for name, param in swm_model.named_parameters() if 'mask' in name}

        lr_decay = True
        if lr_decay:
            lr_scheduler.step()
    
def weight_masking(model, dataset_name, model_file_path, true_target_label, attack_spec, filtered_triggers):
    outer = 100; inner = 5; batch_size = 32
    clean_val_loader, clean_test_loader, inner_iters = prepare_data(dataset_name, batch_size, inner)
    swm_model, criterion, mask_params = initialize_model(model_file_path, dataset_name)

    mask_optimizer = torch.optim.Adam(mask_params, lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=mask_optimizer, gamma=0.9)

    start = time.time()
    trigger = filtered_triggers[0]['Trigger']
    train_test_model(swm_model, criterion, clean_val_loader, clean_test_loader, mask_optimizer, lr_scheduler, outer, inner_iters, trigger)

    print('Running time: {:.4f} s'.format(time.time() - start))
    TargetASR(dataset_name).target_asr(model, swm_model, true_target_label, attack_spec)
