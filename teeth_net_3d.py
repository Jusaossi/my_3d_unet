import torch
import torch.nn as nn
import numpy as np
import os
import platform
import Unet_versions
import torch.optim as optim
from load_my_new_3d_batches import load_my_new_3d_batch
from collections import OrderedDict
from runbuilderclass import RunBuilder
from my_metrics import calculate_my_metrics, calculate_my_sets
import my_loss_functions
from Runmanagerclass import RunManager3D

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

# -----------------load data----------- for now------before dataloader--------------

epoch_numbers = 40
params = OrderedDict(unet=['Unet3D'], loss=['MyDiceBCELoss'], lr=[0.001, 0.0004], albu=['RandomGamma_RandomCrop'], albu_prob=[0.20])

device = torch.device(card)
manager = RunManager3D()
runs_count = 0
np.random.seed(2020)
number_of_batches = 63
test_batches_number = int(np.floor(number_of_batches / 5))
ss = np.random.permutation(number_of_batches) + 1
test_batches = ss[:test_batches_number]
train_batches = ss[test_batches_number:]

for run in RunBuilder.get_runs(params):
    runs_count += 1
    network = getattr(Unet_versions, run.unet)()
    # print(network)

    loss_function = getattr(my_loss_functions, run.loss)()
    network.to(device=device)

    optimizer = optim.Adam(network.parameters(), lr=run.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    manager.begin_run(run)
    my_f1_max_score = 0
    my_f1_score = 0
    for epoch in range(1, epoch_numbers + 1):
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
            print('batch_count=', batch_count, 'batch number =', batch)
            #print('run.batch_size', run.batch_size)
            if batch_count == 5 and machine == 'DESKTOP-K3R0DFP':
                break

            images, _ = load_my_new_3d_batch(batch)
            images = images.astype(np.float32)
            images = torch.as_tensor(images)
            images = images.unsqueeze(0)
            images = images.unsqueeze(0)
            images = images.to(device)

            preds = network(images)

            del images
            _, targets = load_my_new_3d_batch(batch)
            targets = targets.astype(np.float32)
            targets = torch.as_tensor(targets)
            targets = targets.unsqueeze(0)
            targets = targets.unsqueeze(0)
            targets = targets.to(device)


            # print('images shape=', images.shape)
            # print('targets shape=', targets.shape)
            # if run.albu:
            #     images, targets = my_data_albumentations(images, targets, run.albu_prob)
            #     #print('albu megessä')
            # if run.albu != 'no_augmentation':
            # images, targets = my_data_albumentations2(images, targets, run.albu, run.albu_prob)

            # images, targets = my_data_albumentations3(images, targets, run.albu, run.albu_prob)
            # print('images shape=', images.shape)
            # print('targets shape=', targets.shape)



            # print('images shape=', images.shape)
            # print('targets shape=', targets.shape)
            # print(run.alpha)

            #if run.loss == 'MyDiceLoss':
            loss = loss_function(preds, targets)

            print(loss)
            batch_loss = loss.item()
            #manager.track_loss(batch_loss)

            epoch_loss += batch_loss
            # average_epoch_loss = epoch_loss / batch_count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            TP, FP, FN = calculate_my_sets(preds.detach(), targets.detach())
            # print(TP, FP, FN)
            # recall, precision, f1_score,  = calculate_my_metrics(TP, FP, FN)
            # print(recall, precision, f1_score)

            epoch_tp += TP
            epoch_fp += FP
            epoch_fn += FN
            torch.cuda.empty_cache()
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
        #t_epoch_recall = 0
        # t_epoch_true_negative_rate = 0
        #t_epoch_precision = 0
        # t_epoch_accuracy = 0
        #t_epoch_f1_score = 0

        test_epoch_tp = 0
        test_epoch_fp = 0
        test_epoch_fn = 0
        network.eval()
        for test_batch in test_batches:
            test_count += 1
            print('test_count=', test_count, 'testi batch nummero', test_count)
            if test_count == 3 and machine == 'DESKTOP-K3R0DFP':
                break
            images, targets = load_my_new_3d_batch(test_batch)

            images = images.astype(np.float32)
            images = torch.as_tensor(images)
            images = images.unsqueeze(0)
            images = images.unsqueeze(0)
            images = images.to(device)

            targets = targets.astype(np.float32)
            targets = torch.as_tensor(targets)
            targets = targets.unsqueeze(0)
            targets = targets.unsqueeze(0)
            targets = targets.to(device)

            preds = network(images)

            test_loss = loss_function(preds.detach(), targets.detach())
            test_epoch_loss += test_loss.item()

            # t_recall, t_precision, t_f1_score = calculate_my_metrics(preds.detach(), targets.detach())
            # t_epoch_recall += t_recall
            # # t_epoch_true_negative_rate += t_true_negative_rate
            # t_epoch_precision += t_precision
            # # t_epoch_accuracy += t_accuracy
            # t_epoch_f1_score += t_f1_score
            test_TP, test_FP, test_FN = calculate_my_sets(preds.detach(), targets.detach())

            test_epoch_tp += test_TP
            test_epoch_fp += test_FP
            test_epoch_fn += test_FN
#             # if test_batch % 1 == 0:
#             #     with open(save_file_new_2, 'a', newline='') as f:
#             #         result = [runs_count, run.unet, run.loss, epoch, test_count, run.data, test_patient, str(test_slices),
#             #                   run.lr, test_batch_size, round(test_loss.item(), 4),
#             #                   round(test_batch_l1_loss, 4), test_batch_correct_teeth, test_batch_teeth_all]
#             #
#             #         writer = csv.writer(f)
#             #         writer.writerow(result)
            torch.cuda.empty_cache()
        epoch_test_recall = test_epoch_tp / (test_epoch_tp + test_epoch_fn)
        epoch_test_precision = test_epoch_tp / (test_epoch_tp + test_epoch_fp)
        epoch_test_f1_score = (2 * epoch_test_precision * epoch_test_recall) / (epoch_test_precision + epoch_test_recall)
        print('paskaa 2', epoch_test_recall, epoch_test_precision, epoch_test_f1_score)

        manager.track_test_loss(test_epoch_loss, test_count)
        # manager.track_test_num_correct(t_epoch_recall, t_epoch_precision, t_epoch_f1_score)
        manager.track_test_true_epoch_metrics(epoch_test_recall, epoch_test_precision, epoch_test_f1_score)

        # if epoch_test_f1_score > my_f1_score:
        #     my_f1_score = epoch_test_f1_score
        #     print('model now save, epoch =', epoch)
        #     print('epoch_test_f1_score:', epoch_test_f1_score)
        #     torch.save(network, my_save_path + '\\' + 'two_augh_network_60_epoch.pth')
        # # scheduler.step()
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