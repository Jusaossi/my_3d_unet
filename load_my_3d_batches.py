import numpy as np
import os
import matplotlib.pyplot as plt
import platform

def load_my_3d_batch(my_batch):
    machine = platform.node()
    batch_nro = my_batch
    cube_size = 64
    andy_cut = [470, 620, 120, 250, 250, 410]

    andy_1 = [andy_cut[0], andy_cut[0] + cube_size, andy_cut[2], andy_cut[2] + cube_size, andy_cut[4] + 20, andy_cut[4] + 20 + cube_size]
    andy_2 = [andy_cut[0], andy_cut[0] + cube_size, andy_cut[2], andy_cut[2] + cube_size, andy_cut[4] + 48, andy_cut[4] + 48 + cube_size]
    andy_3 = [andy_cut[0], andy_cut[0] + cube_size, andy_cut[2], andy_cut[2] + cube_size, andy_cut[5] - 20 - cube_size, andy_cut[5] - 20]
    andy_4 = [andy_cut[0], andy_cut[0] + cube_size, andy_cut[2] + 33, andy_cut[2] + 33 + cube_size, andy_cut[4], andy_cut[4] + cube_size]
    andy_5 = [andy_cut[0], andy_cut[0] + cube_size, andy_cut[2] + 33, andy_cut[2] + 33 + cube_size, andy_cut[5] - cube_size, andy_cut[5]]
    andy_6 = [andy_cut[0], andy_cut[0] + cube_size, andy_cut[3] - cube_size, andy_cut[3], andy_cut[4], andy_cut[4] + cube_size]
    andy_7 = [andy_cut[0], andy_cut[0] + cube_size, andy_cut[3] - cube_size, andy_cut[3], andy_cut[5] - cube_size, andy_cut[5]]

    andy_8 = [andy_cut[0] + 43, andy_cut[0] + 43 + cube_size, andy_cut[2], andy_cut[2] + cube_size, andy_cut[4] + 20, andy_cut[4] + 20 + cube_size]
    andy_9 = [andy_cut[0] + 43, andy_cut[0] + 43 + cube_size, andy_cut[2], andy_cut[2] + cube_size, andy_cut[4] + 48, andy_cut[4] + 48 + cube_size]
    andy_10 = [andy_cut[0] + 43, andy_cut[0] + 43 + cube_size, andy_cut[2], andy_cut[2] + cube_size, andy_cut[5] - 20 - cube_size, andy_cut[5] - 20]
    andy_11 = [andy_cut[0] + 43, andy_cut[0] + 43 + cube_size, andy_cut[2] + 33, andy_cut[2] + 33 + cube_size, andy_cut[4], andy_cut[4] + cube_size]
    andy_12 = [andy_cut[0] + 43, andy_cut[0] + 43 + cube_size, andy_cut[2] + 33, andy_cut[2] + 33 + cube_size, andy_cut[5] - cube_size, andy_cut[5]]
    andy_13 = [andy_cut[0] + 43, andy_cut[0] + 43 + cube_size, andy_cut[3] - cube_size, andy_cut[3], andy_cut[4], andy_cut[4] + cube_size]
    andy_14 = [andy_cut[0] + 43, andy_cut[0] + 43 + cube_size, andy_cut[3] - cube_size, andy_cut[3], andy_cut[5] - cube_size, andy_cut[5]]

    andy_15 = [andy_cut[1] - cube_size, andy_cut[1], andy_cut[2], andy_cut[2] + cube_size, andy_cut[4] + 20, andy_cut[4] + 20 + cube_size]
    andy_16 = [andy_cut[1] - cube_size, andy_cut[1], andy_cut[2], andy_cut[2] + cube_size, andy_cut[4] + 48, andy_cut[4] + 48 + cube_size]
    andy_17 = [andy_cut[1] - cube_size, andy_cut[1], andy_cut[2], andy_cut[2] + cube_size, andy_cut[5] - 20 - cube_size, andy_cut[5] - 20]
    andy_18 = [andy_cut[1] - cube_size, andy_cut[1], andy_cut[2] + 33, andy_cut[2] + 33 + cube_size, andy_cut[4], andy_cut[4] + cube_size]
    andy_19 = [andy_cut[1] - cube_size, andy_cut[1], andy_cut[2] + 33, andy_cut[2] + 33 + cube_size, andy_cut[5] - cube_size, andy_cut[5]]
    andy_20 = [andy_cut[1] - cube_size, andy_cut[1], andy_cut[3] - cube_size, andy_cut[3], andy_cut[4], andy_cut[4] + cube_size]
    andy_21 = [andy_cut[1] - cube_size, andy_cut[1], andy_cut[3] - cube_size, andy_cut[3], andy_cut[5] - cube_size, andy_cut[5]]


    my_batches = {1: ['andy', andy_1], 2: ['andy', andy_2], 3: ['andy', andy_3], 4: ['andy', andy_4], 5: ['andy', andy_5],
                  6: ['andy', andy_6], 7: ['andy', andy_7], 8: ['andy', andy_8], 9: ['andy', andy_9], 10: ['andy', andy_10],
                  11: ['andy', andy_11], 12: ['andy', andy_12], 13: ['andy', andy_13], 14: ['andy', andy_14],
                  15: ['andy', andy_15],
                  16: ['andy', andy_16], 17: ['andy', andy_17], 18: ['andy', andy_18], 19: ['andy', andy_19],
                  20: ['andy', andy_20], 21: ['andy', andy_21]}
    patient = my_batches[batch_nro][0]
    my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = my_batches[batch_nro][1]
    if machine == 'DESKTOP-K3R0DFP':
        my_path = r'C:\Users\jpkorpel\Desktop\hammas'
    else:
        my_dir = os.getcwd()
        my_path = os.path.join(my_dir, 'hammas')

    patient_images_file = 'X_images_' + patient + '.npy'
    patient_targets_file = 'Y_targets_' + patient + '.npy'
    image_folder = os.path.join(my_path, patient, patient_images_file)
    target_folder = os.path.join(my_path, patient, patient_targets_file)
    #print(image_folder)
    #print(target_folder)
    # ------------------------------andy------------------------------------------------------------------------
    #X1 = np.load(image_folder)[470:620, 120:250, 250:410]
    #Y1 = np.load(r'C:\Users\jpkorpel\Desktop\hammas\andy\Y_targets_andy.npy')[470:620, 120:250, 250:410]
    #apu2 = np.load(image_folder)[470:534, 120:184, 270:334]
    my_images = np.load(image_folder)[my_slicer1: my_slicer2, my_slicer3: my_slicer4, my_slicer5: my_slicer6]
    my_targets = np.load(target_folder)[my_slicer1: my_slicer2, my_slicer3: my_slicer4, my_slicer5: my_slicer6]
    return my_images, my_targets
# exit()
# X1_eka = X1[:64, :64, 20: 84]
# X1_toka = X1[:64, :64, int(X1.shape[2] / 2) - 32: int(X1.shape[2] / 2) + 32]
# X1_kolmas = X1[:64, :64, -84: -20]
# X1_neljas = X1[:64, int(X1.shape[1] / 2) - 32: int(X1.shape[1] / 2) + 32, :64]
# X1_viides = X1[:64, int(X1.shape[1] / 2) - 32: int(X1.shape[1] / 2) + 32, -64:]
# X1_kuudes = X1[:64, -64:, :64]
# X1_seiska = X1[:64, -64:, -64:]
# print(X1.shape)
# print(X1_eka.shape)
# print(X1_toka.shape)
# print(X1_kolmas.shape)
# print(X1_neljas.shape)
# print(X1_viides.shape)
# print(X1_kuudes.shape)
# print(X1_seiska.shape)
#
# print(X1.shape[2] - 20 - (X1.shape[2] - 84))
# print(X1.shape[2] - 20, X1.shape[2] - 84)
# s = 63
# plt.figure(10)
# plt.suptitle('andy, blokit x-y tasossa, z=63')
# plt.subplot(3, 3, 1)
# plt.imshow(X1_eka[s])
# plt.subplot(3, 3, 2)
# plt.imshow(X1_toka[s])
# plt.subplot(3, 3, 3)
# plt.imshow(X1_kolmas[s])
# plt.subplot(3, 3, 4)
# plt.imshow(X1_neljas[s])
# plt.subplot(3, 3, 6)
# plt.imshow(X1_viides[s])
# plt.subplot(3, 3, 7)
# plt.imshow(X1_kuudes[s])
# plt.subplot(3, 3, 8)
# plt.imshow(apu[s])
# plt.subplot(3, 3, 9)
# plt.imshow(X1_seiska[s])
# plt.show()
# exit()
#
#
# for j in range(0, 10):
#     plt.figure(j)
#     plt.title(15 * j + 7)
#     plt.subplot(1, 2, 1)
#     plt.imshow(X1[15 * j + 7])
#     # ---------------ylä eka---------------------------------------------
#     plt.plot(np.arange(20, 84), 63 * np.ones(64), color='y')
#     plt.plot(20 * np.ones(64), np.arange(64), color='y')
#     plt.plot(83 * np.ones(64), np.arange(64), color='y')
# # ------------------ ylä vika ------------------------------------
#     plt.plot(np.arange(X1.shape[2] - 84, X1.shape[2] - 20), 63 * np.ones(64), color='b')
#     plt.plot((X1.shape[2] - 21) * np.ones(64), np.arange(64), color='b')
#     plt.plot((X1.shape[2] - 84) * np.ones(64), np.arange(64), color='b')
#     # ------------------ ylä keski ------------------------------------
#     plt.plot(np.arange(X1.shape[2] / 2 - 32, X1.shape[2] / 2 + 32), 63 * np.ones(64), color='w')
#     plt.plot((X1.shape[2] / 2 - 32) * np.ones(64), np.arange(64), color='w')
#     plt.plot((X1.shape[2] / 2 + 31) * np.ones(64), np.arange(64), color='w')
#
# #------------------- ala eka -------------------------------------------------
#     plt.plot(63 * np.ones(64), np.arange(X1.shape[1]-64, X1.shape[1]), color='r')
#     plt.plot(np.arange(64), (X1.shape[1]-64) * np.ones(64), color='r')
#     # ------------------- ala eka keski -------------------------------------------------
#     plt.plot(63 * np.ones(64), np.arange(X1.shape[1] / 2 - 32, X1.shape[1] / 2 + 32), color='gray')
#     plt.plot(np.arange(64), (X1.shape[1] / 2 - 32) * np.ones(64), color='gray')
#     plt.plot(np.arange(64), (X1.shape[1] / 2 + 31) * np.ones(64), color='gray')
#     # ------------------- ala vika -------------------------------------------------
#     plt.plot(np.arange(X1.shape[2] - 64, X1.shape[2]), (X1.shape[1] - 64) * np.ones(64), color='r')
#     plt.plot((X1.shape[2] - 64) * np.ones(64), np.arange(X1.shape[1] - 64, X1.shape[1]), color='r')
#     # ------------------- ala vika keski -------------------------------------------------
#     plt.plot((X1.shape[2] - 64) * np.ones(64), np.arange(X1.shape[1] / 2 - 32, X1.shape[1] / 2 + 32), color='gray')
#     plt.plot(np.arange(X1.shape[2] - 64, X1.shape[2]), (X1.shape[1] / 2 - 32) * np.ones(64), color='gray')
#     plt.plot(np.arange(X1.shape[2] - 64, X1.shape[2]), (X1.shape[1] / 2 + 31) * np.ones(64), color='gray')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(Y1[15 * j + 7])
#     # ---------------ylä eka---------------------------------------------
#     plt.plot(np.arange(20, 84), 63 * np.ones(64), color='y')
#     plt.plot(20 * np.ones(64), np.arange(64), color='y')
#     plt.plot(83 * np.ones(64), np.arange(64), color='y')
#     # ------------------ ylä vika ------------------------------------
#     plt.plot(np.arange(X1.shape[2] - 84, X1.shape[2] - 20), 63 * np.ones(64), color='b')
#     plt.plot((X1.shape[2] - 21) * np.ones(64), np.arange(64), color='b')
#     plt.plot((X1.shape[2] - 84) * np.ones(64), np.arange(64), color='b')
#     # ------------------ ylä keski ------------------------------------
#     plt.plot(np.arange(X1.shape[2] / 2 - 32, X1.shape[2] / 2 + 32), 63 * np.ones(64), color='w')
#     plt.plot((X1.shape[2] / 2 - 32) * np.ones(64), np.arange(64), color='w')
#     plt.plot((X1.shape[2] / 2 + 31) * np.ones(64), np.arange(64), color='w')
#
#     # ------------------- ala eka -------------------------------------------------
#     plt.plot(63 * np.ones(64), np.arange(X1.shape[1] - 64, X1.shape[1]), color='r')
#     plt.plot(np.arange(64), (X1.shape[1] - 64) * np.ones(64), color='r')
#     # ------------------- ala eka keski -------------------------------------------------
#     plt.plot(63 * np.ones(64), np.arange(X1.shape[1] / 2 - 32, X1.shape[1] / 2 + 32), color='gray')
#     plt.plot(np.arange(64), (X1.shape[1] / 2 - 32) * np.ones(64), color='gray')
#     plt.plot(np.arange(64), (X1.shape[1] / 2 + 31) * np.ones(64), color='gray')
#     # ------------------- ala vika -------------------------------------------------
#     plt.plot(np.arange(X1.shape[2] - 64, X1.shape[2]), (X1.shape[1] - 64) * np.ones(64), color='r')
#     plt.plot((X1.shape[2] - 64) * np.ones(64), np.arange(X1.shape[1] - 64, X1.shape[1]), color='r')
#     # ------------------- ala vika keski -------------------------------------------------
#     plt.plot((X1.shape[2] - 64) * np.ones(64), np.arange(X1.shape[1] / 2 - 32, X1.shape[1] / 2 + 32), color='gray')
#     plt.plot(np.arange(X1.shape[2] - 64, X1.shape[2]), (X1.shape[1] / 2 - 32) * np.ones(64), color='gray')
#     plt.plot(np.arange(X1.shape[2] - 64, X1.shape[2]), (X1.shape[1] / 2 + 31) * np.ones(64), color='gray')
# plt.show()