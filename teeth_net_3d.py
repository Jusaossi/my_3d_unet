import torch
import torch.nn as nn
import numpy as np
import os
import platform
import Unet_versions
import torch.optim as optim
from load_my_new_3d_batches import load_my_new_3d_batch, load_my_target
from collections import OrderedDict
from runbuilderclass import RunBuilder
from my_metrics import calculate_my_metrics, calculate_my_sets
import my_loss_functions
from Runmanagerclass import RunManager3D


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(file):
    global my_epoch
    global my_f1_score
    print('=> Loading checkpoint')
    my_checkpoint = torch.load(file)
    network.load_state_dict(my_checkpoint['state_dict'])
    optimizer.load_state_dict(my_checkpoint['optimizer'])
    my_epoch = my_checkpoint['Epoch']
    my_f1_score = my_checkpoint['F1_score']


load_model = False
machine = platform.node()
# my_data_dir = None
if machine == 'DESKTOP-K3R0DFP':
    my_save_path_help = r'C:\Users\jpkorpel\PycharmProjects\my_first_net'
    my_save_path = os.path.join(my_save_path_help, 'runs')
    card = 'cpu'
    snapshot_path = r'C:\Users\jpkorpel\PycharmProjects\my_3d_unet'
else:
    my_parent_dir = os.path.dirname(os.getcwd())
    my_save_path = os.path.join(my_parent_dir, 'my_3d_unet')
    card = 'cuda'

# -----------------load data----------- for now------before dataloader--------------, 'random_small' 'random_same'

epoch_numbers = 150
params = OrderedDict(unet=['Unet3D'], loss=['MyDiceBCELoss'], lr=[0.00008], scale=['[0,1]'], crop_cube=[64], crop_strategy=['no_crop'])

device = torch.device(card)
manager = RunManager3D()
runs_count = 0
np.random.seed(2021)
number_of_batches = 252
test_batches_number = int(np.floor(number_of_batches / 5))
ss = np.random.permutation(number_of_batches) + 1
test_batches = ss[:test_batches_number]
train_batches = ss[test_batches_number:]
# test_batches = range(190, 211)  # (211, 232)  tai (232, 253)  .. uusille seteille

for run in RunBuilder.get_runs(params):
    runs_count += 1
    network = getattr(Unet_versions, run.unet)()
    #print(network)

    loss_function = getattr(my_loss_functions, run.loss)()
    network.to(device=device)

    optimizer = optim.Adam(network.parameters(), lr=run.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    my_epoch = 0
    my_f1_score = 0
    if load_model:
        load_checkpoint('my_checkpoint.pth.tar')
    manager.begin_run(run, my_epoch)
    # my_f1_max_score = 0

    for epoch in range(my_epoch + 1, epoch_numbers + my_epoch + 1):
        network.train()
        print(f'run = {runs_count}, epoch = {epoch}')
        manager.begin_epoch()
        batch_count = 0
        epoch_loss = 0
        epoch_tp = 0
        epoch_fp = 0
        epoch_fn = 0

        for batch in train_batches:
            batch_count += 1
            # print('batch_count=', batch_count, 'batch number =', batch)
            # print('run.batch_size', run.batch_size)
            if batch_count == 5 and machine == 'DESKTOP-K3R0DFP':
                break

            images, target_folder, my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = \
                load_my_new_3d_batch(batch, run.scale, lower_cut=200, crop_cube=run.crop_cube, crop_strategy=run.crop_strategy)

            images = torch.as_tensor(images)
            images = images.unsqueeze(0)
            images = images.unsqueeze(0)
            images = images.to(device)
            preds = network(images)
            del images

            targets = load_my_target(target_folder, my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6)
            targets = targets.astype(np.float32)
            targets = torch.as_tensor(targets)
            targets = targets.unsqueeze(0)
            targets = targets.unsqueeze(0)
            targets = targets.to(device)

            loss = loss_function(preds, targets)

            batch_loss = loss.detach().item()

            epoch_loss += batch_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            TP, FP, FN = calculate_my_sets(preds.detach(), targets.detach())

            epoch_tp += TP
            epoch_fp += FP
            epoch_fn += FN
            torch.cuda.empty_cache()
            del targets
        epoch_train_recall = epoch_tp / (epoch_tp + epoch_fn)
        epoch_train_precision = epoch_tp / (epoch_tp + epoch_fp)
        epoch_train_f1_score = (2 * epoch_train_precision * epoch_train_recall) / (epoch_train_precision + epoch_train_recall)
        # print('paskaa', epoch_train_recall, epoch_train_precision, epoch_train_f1_score)
        #manager.track_test_loss(test_epoch_loss, test_count)
        manager.track_train_epoch_metrics(epoch_train_recall, epoch_train_precision, epoch_train_f1_score)
        manager.track_train_loss(epoch_loss, batch_count)
        #manager.track_test_epoch_metrics(epoch_tp, epoch_fp, epoch_fn)
        # torch.cuda.empty_cache()
        test_count = 0
        test_epoch_loss = 0

        test_epoch_tp = 0
        test_epoch_fp = 0
        test_epoch_fn = 0
        network.eval()
        for test_batch in test_batches:
            test_count += 1
            #print('test_count=', test_count, 'testi batch nummero', test_count)
            if test_count == 3 and machine == 'DESKTOP-K3R0DFP':
                break
            images, target_folder, my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = \
                load_my_new_3d_batch(test_batch, run.scale, lower_cut=200, crop_cube='False', crop_strategy=run.crop_strategy)
            images = images.astype(np.float32)
            images = torch.as_tensor(images)
            images = images.unsqueeze(0)
            images = images.unsqueeze(0)
            images = images.to(device)

            preds = network(images)
            del images
            targets = load_my_target(target_folder, my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5,
                                     my_slicer6)
            targets = targets.astype(np.float32)
            targets = torch.as_tensor(targets)
            targets = targets.unsqueeze(0)
            targets = targets.unsqueeze(0)
            targets = targets.to(device)

            test_loss = loss_function(preds.detach(), targets.detach())
            test_epoch_loss += test_loss.item()

            test_TP, test_FP, test_FN = calculate_my_sets(preds.detach(), targets.detach())

            test_epoch_tp += test_TP
            test_epoch_fp += test_FP
            test_epoch_fn += test_FN
            torch.cuda.empty_cache()
            del targets

        epoch_test_recall = test_epoch_tp / (test_epoch_tp + test_epoch_fn)
        epoch_test_precision = test_epoch_tp / (test_epoch_tp + test_epoch_fp)
        epoch_test_f1_score = (2 * epoch_test_precision * epoch_test_recall) / (epoch_test_precision + epoch_test_recall)
        print('test results=', epoch_test_recall, epoch_test_precision, epoch_test_f1_score)

        manager.track_test_loss(test_epoch_loss, test_count)
        # manager.track_test_num_correct(t_epoch_recall, t_epoch_precision, t_epoch_f1_score)
        manager.track_test_true_epoch_metrics(epoch_test_recall, epoch_test_precision, epoch_test_f1_score)

        # if epoch_test_f1_score > my_f1_score:
        #     my_f1_score = epoch_test_f1_score
        #     print('model now save, epoch =', epoch)
        #     print('epoch_test_f1_score:', epoch_test_f1_score)
        #     torch.save(network, my_save_path + '\\' + '3d_network_260_epoch_own_scale_200_clamp.pth')
        # # scheduler.step()
        if epoch % 5 == 0:
            checkpoint = {'state_dict': network.state_dict(), 'optimizer': optimizer.state_dict(), 'Epoch': epoch, 'F1_score': my_f1_score}
            save_checkpoint(checkpoint)
        manager.end_epoch()
        torch.cuda.empty_cache()

    manager.end_run()
# # manager.save(save_file)
#
#
# for j in range(1):
#     preds = network(images)
#     print('preds shape', preds.shape)
#     loss_function = nn.BCELoss()
#     loss = loss_function(preds, targets)
#     print(loss)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()