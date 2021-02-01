import numpy as np
import os
import matplotlib.pyplot as plt
import platform

machine = platform.node()
plt.figure(10)


def load_my_new_3d_batch(batch_nro):

    my_batches = {1: ['andy', 'batch_1'], 2: ['andy', 'batch_2'], 3: ['andy', 'batch_3'], 4: ['andy', 'batch_4'],
                  5: ['andy', 'batch_5'], 6: ['andy', 'batch_6'], 7: ['andy', 'batch_7'], 8: ['andy', 'batch_8'],
                  9: ['andy', 'batch_9'], 10: ['andy', 'batch_10'], 11: ['andy', 'batch_11'], 12: ['andy', 'batch_12'],
                  13: ['andy', 'batch_13'], 14: ['andy', 'batch_14'], 15: ['andy', 'batch_15'], 16: ['andy', 'batch_16'],
                  17: ['andy', 'batch_17'], 18: ['andy', 'batch_18'], 19: ['andy', 'batch_19'], 20: ['andy', 'batch_20'],
                  21: ['andy', 'batch_21'],
                  22: ['teeth1', 'batch_1'], 23: ['teeth1', 'batch_2'], 24: ['teeth1', 'batch_3'], 25: ['teeth1', 'batch_4'],
                  26: ['teeth1', 'batch_5'], 27: ['teeth1', 'batch_6'], 28: ['teeth1', 'batch_7'], 29: ['teeth1', 'batch_8'],
                  30: ['teeth1', 'batch_9'], 31: ['teeth1', 'batch_10'],32: ['teeth1', 'batch_11'], 33: ['teeth1', 'batch_12'],
                  34: ['teeth1', 'batch_13'], 35: ['teeth1', 'batch_14'], 36: ['teeth1', 'batch_15'], 37: ['teeth1', 'batch_16'],
                  38: ['teeth1', 'batch_17'], 39: ['teeth1', 'batch_18'], 40: ['teeth1', 'batch_19'], 41: ['teeth1', 'batch_20'],
                  42: ['teeth1', 'batch_21'],
                  43: ['teeth2', 'batch_1'], 44: ['teeth2', 'batch_2'], 45: ['teeth2', 'batch_3'], 46: ['teeth2', 'batch_4'],
                  47: ['teeth2', 'batch_5'], 48: ['teeth2', 'batch_6'], 49: ['teeth2', 'batch_7'], 50: ['teeth2', 'batch_8'],
                  51: ['teeth2', 'batch_9'], 52: ['teeth2', 'batch_10'], 53: ['teeth2', 'batch_11'], 54: ['teeth2', 'batch_12'],
                  55: ['teeth2', 'batch_13'], 56: ['teeth2', 'batch_14'], 57: ['teeth2', 'batch_15'], 58: ['teeth2', 'batch_16'],
                  59: ['teeth2', 'batch_17'], 60: ['teeth2', 'batch_18'], 61: ['teeth2', 'batch_19'], 62: ['teeth2', 'batch_20'],
                  63: ['teeth2', 'batch_21'],
                  64: ['patient1', 'batch_1'], 65: ['patient1', 'batch_2'], 66: ['patient1', 'batch_3'],
                  67: ['patient1', 'batch_4'], 68: ['patient1', 'batch_5'], 69: ['patient1', 'batch_6'], 70: ['patient1', 'batch_7'],
                  71: ['patient1', 'batch_8'], 72: ['patient1', 'batch_9'], 73: ['patient1', 'batch_10'], 74: ['patient1', 'batch_11'],
                  75: ['patient1', 'batch_12'], 76: ['patient1', 'batch_13'], 77: ['patient1', 'batch_14'], 78: ['patient1', 'batch_15'],
                  79: ['patient1', 'batch_16'], 80: ['patient1', 'batch_17'], 81: ['patient1', 'batch_18'], 82: ['patient1', 'batch_19'],
                  83: ['patient1', 'batch_20'], 84: ['patient1', 'batch_21'], 85: ['patient1', 'batch_22'], 86: ['patient1', 'batch_23'],
                  87: ['patient1', 'batch_24'], 88: ['patient1', 'batch_25'], 89: ['patient1', 'batch_26'], 90: ['patient1', 'batch_27'],
                  91: ['patient1', 'batch_28'], 92: ['patient1', 'batch_29'], 93: ['patient1', 'batch_30'], 94: ['patient1', 'batch_31'],
                  95: ['patient1', 'batch_32'], 96: ['patient1', 'batch_33'], 97: ['patient1', 'batch_34'],
                  98: ['patient1', 'batch_35'],
                  99: ['timo', 'batch_1'], 100: ['timo', 'batch_2'], 101: ['timo', 'batch_3'], 102: ['timo', 'batch_4'],
                  103: ['timo', 'batch_5'], 104: ['timo', 'batch_6'], 105: ['timo', 'batch_7'], 106: ['timo', 'batch_8'],
                  107: ['timo', 'batch_9'], 108: ['timo', 'batch_10'], 109: ['timo', 'batch_11'], 110: ['timo', 'batch_12'],
                  111: ['timo', 'batch_13'], 112: ['timo', 'batch_14'], 113: ['timo', 'batch_15'], 114: ['timo', 'batch_16'],
                  115: ['timo', 'batch_17'], 116: ['timo', 'batch_18'], 117: ['timo', 'batch_19'], 118: ['timo', 'batch_20'],
                  119: ['timo', 'batch_21'],
                  }

    patient = my_batches[batch_nro][0]

    batch_cut = None
    x_half_length = None
    y_half_length = None
    z_half_length = None
    z_quarter_length = 1
    cube_size = None
    cube_size_z = 1

    if patient == 'andy':
        batch_cut = [470, 620, 120, 250, 250, 410]
        x_half_length = int(160 / 2)
        y_half_length = int(130 / 2)
        z_half_length = int(150 / 2)
        cube_size = 64
    elif patient == 'teeth1':
        batch_cut = [30, 194, 50, 280, 100, 350]
        x_half_length = int(250 / 2)    # 70 mimimi
        y_half_length = int(230 / 2)   # 77 minimi
        z_half_length = int(164 / 2)
        cube_size = 96
    elif patient == 'teeth2':
        batch_cut = [150, 306, 20, 210, 150, 390]
        x_half_length = int(240 / 2)  # 66,66 minimi
        y_half_length = int(190 / 2)  # 63,33
        z_half_length = int(156 / 2)
        cube_size = 88
    elif patient == 'timo':
        batch_cut = [440, 600, 80, 230, 240, 440]
        x_half_length = int(200 / 2)  # 66,66 minimi
        y_half_length = int(150 / 2)  # 63,33
        z_half_length = int(160 / 2)
        cube_size = 72

    elif patient == 'patient1':
        batch_cut = [40, 420, 26, 410, 30, 514]
        x_half_length = int(500 / 2)  # 66,66 minimi
        y_half_length = int(400 / 2)  # 63,33
        z_half_length = int(380 / 2)
        z_quarter_length = int(380 / 4)
        cube_size = 160
        cube_size_z = 80
    #print(patient, cube_size)
    # print(my_batches[batch_nro][1])

    # batch_coordinates ={'batch_1': [batch_cut[0], batch_cut[0] + cube_size, batch_cut[2], batch_cut[2] + cube_size, batch_cut[4] + 20, batch_cut[4] + 20 + cube_size]}


    # my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = my_batches[batch_nro][1]


    # andy_cut = [470, 620, 120, 250, 250, 410]
    batch_coordinates = {
        'batch_1': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + 30, batch_cut[4] + 30 + cube_size],
        'batch_2': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_half_length - int(cube_size / 2), batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_3': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[5] - 30 - cube_size, batch_cut[5] - 30],
        'batch_4': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2] + y_half_length - int(cube_size / 2), batch_cut[2] + y_half_length + int(cube_size / 2),
                    batch_cut[4] + 15, batch_cut[4] + 15 + cube_size],
        'batch_5': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2] + y_half_length - int(cube_size / 2), batch_cut[2] + y_half_length + int(cube_size / 2),
                    batch_cut[5] - 15 - cube_size, batch_cut[5] - 15],
        'batch_6': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[3] - cube_size, batch_cut[3],
                    batch_cut[4], batch_cut[4] + cube_size],
        'batch_7': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[3] - cube_size, batch_cut[3],
                    batch_cut[5] - cube_size, batch_cut[5]],
        'batch_8': [batch_cut[0] + z_half_length - int(cube_size / 2), batch_cut[0] + z_half_length + int(cube_size / 2),
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + 30, batch_cut[4] + 30 + cube_size],
        'batch_9': [batch_cut[0] + z_half_length - int(cube_size / 2), batch_cut[0] + z_half_length + int(cube_size / 2),
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_half_length - int(cube_size / 2), batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_10': [batch_cut[0] + z_half_length - int(cube_size / 2), batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - 30 - cube_size, batch_cut[5] - 30],
        'batch_11': [batch_cut[0] + z_half_length - int(cube_size / 2), batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2), batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + 15, batch_cut[4] + 15 + cube_size],
        'batch_12': [batch_cut[0] + z_half_length - int(cube_size / 2), batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2), batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - 15 - cube_size, batch_cut[5] - 15],
        'batch_13': [batch_cut[0] + z_half_length - int(cube_size / 2), batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[3] - cube_size,  batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_14': [batch_cut[0] + z_half_length - int(cube_size / 2), batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],
        'batch_15': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + 30, batch_cut[4] + 30 + cube_size],
        'batch_16': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_half_length - int(cube_size / 2), batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_17': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - 30 - cube_size, batch_cut[5] - 30],
        'batch_18': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2), batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + 15, batch_cut[4] + 15 + cube_size],
        'batch_19': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2), batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - 15 - cube_size, batch_cut[5] - 15],
        'batch_20': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_21': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],
    }

    batch_coordinates2 = {

        'batch_1': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + 60, batch_cut[4] + 60 + cube_size],
        'batch_2': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_half_length - int(cube_size / 2),
                    batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_3': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[5] - 60 - cube_size, batch_cut[5] - 60],
        'batch_4': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2] + y_half_length - int(cube_size / 2),
                    batch_cut[2] + y_half_length + int(cube_size / 2),
                    batch_cut[4] + 30, batch_cut[4] + 30 + cube_size],
        'batch_5': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2] + y_half_length - int(cube_size / 2),
                    batch_cut[2] + y_half_length + int(cube_size / 2),
                    batch_cut[5] - 30 - cube_size, batch_cut[5] - 30],
        'batch_6': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[3] - cube_size, batch_cut[3],
                    batch_cut[4], batch_cut[4] + cube_size],
        'batch_7': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[3] - cube_size, batch_cut[3],
                    batch_cut[5] - cube_size, batch_cut[5]],

        'batch_8': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                    batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + 60, batch_cut[4] + 60 + cube_size],
        'batch_9': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                    batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_half_length - int(cube_size / 2),
                    batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_10': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - 60 - cube_size, batch_cut[5] - 60],
        'batch_11': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + 30, batch_cut[4] + 30 + cube_size],
        'batch_12': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - 30 - cube_size, batch_cut[5] - 30],
        'batch_13': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_14': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_15': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                    batch_cut[0] + z_half_length + int(cube_size_z / 2),
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + 60, batch_cut[4] + 60 + cube_size],
        'batch_16': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                    batch_cut[0] + z_half_length + int(cube_size_z / 2),
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_half_length - int(cube_size / 2),
                    batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_17': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - 60 - cube_size, batch_cut[5] - 60],
        'batch_18': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + 30, batch_cut[4] + 30 + cube_size],
        'batch_19': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - 30 - cube_size, batch_cut[5] - 30],
        'batch_20': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_21': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_22': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + 60, batch_cut[4] + 60 + cube_size],
        'batch_23': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_half_length - int(cube_size / 2),
                     batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_24': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - 60 - cube_size, batch_cut[5] - 60],
        'batch_25': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + 30, batch_cut[4] + 30 + cube_size],
        'batch_26': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - 30 - cube_size, batch_cut[5] - 30],
        'batch_27': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_28': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_29': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + 60, batch_cut[4] + 60 + cube_size],
        'batch_30': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_half_length - int(cube_size / 2),
                     batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_31': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - 60 - cube_size, batch_cut[5] - 60],
        'batch_32': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + 30, batch_cut[4] + 30 + cube_size],
        'batch_33': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - 30 - cube_size, batch_cut[5] - 30],
        'batch_34': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_35': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]]
    }
    # print(batch_coordinates[my_batches[batch_nro][1]])
    #my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = batch_coordinates[my_batches[batch_nro + 0][1]]
    # print(my_slicer1)
    if patient != 'patient1':
        my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = batch_coordinates[my_batches[batch_nro][1]]
        
    else:
        print('mulkku')
        my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = batch_coordinates2[my_batches[batch_nro][1]]

    if machine == 'DESKTOP-K3R0DFP':
        my_path = r'C:\Users\jpkorpel\Desktop\hammas'
    else:
        my_dir = os.getcwd()
        my_path = os.path.join(my_dir, 'hammas')
    #patient = 'patient1'
    patient_images_file = 'X_images_' + patient + '.npy'
    patient_targets_file = 'Y_targets_' + patient + '.npy'
    image_folder = os.path.join(my_path, patient, patient_images_file)
    #print(image_folder)
    target_folder = os.path.join(my_path, patient, patient_targets_file)

    # plt.figure(11)
    # for j in range(5):
    #     print(j)
    #     my_images = np.load(image_folder)[40 + j * 75 : 40 + j * 75 + 80, 140, 30: 514]
    #     print(j,'on',  [40 + j * 75, 40 + j * 75 + 80])
    #     my_targets = np.load(target_folder)[40 + j * 75 : 40 + j * 75 + 80, 140, 30: 514]
    #     print(my_targets.shape)
    #     plt.subplot(5, 2, 2 * j + 1)
    #     plt.imshow(my_images)
    #     plt.subplot(5, 2, 2 * j + 2)
    #     plt.imshow(my_targets)
    #
    # plt.show()
    # exit()
    my_images = np.load(image_folder)[my_slicer1: my_slicer2, my_slicer3: my_slicer4, my_slicer5: my_slicer6]
    my_targets = np.load(target_folder)[my_slicer1: my_slicer2, my_slicer3: my_slicer4, my_slicer5: my_slicer6]
    return my_images, my_targets
#     print(my_images.shape)
#     print(my_targets.shape)
#     s = 40
#     print('my j', j)
#
#     if j in [1, 2, 3]:
#         plt.subplot(3, 6, j)
#         plt.imshow(my_images[s])
#         plt.subplot(3, 6, j + 3)
#         plt.imshow(my_targets[s])
#     elif j == 4:
#         plt.subplot(3, 6, j + 3)
#         plt.imshow(my_images[s])
#         plt.subplot(3, 6, j + 6)
#         plt.imshow(my_targets[s])
#     elif j == 5:
#         plt.subplot(3, 6, j + 4)
#         plt.imshow(my_images[s])
#         plt.subplot(3, 6, j + 7)
#         plt.imshow(my_targets[s])
#     elif j == 6:
#         plt.subplot(3, 6, j + 7)
#         plt.imshow(my_images[s])
#         plt.subplot(3, 6, j + 10)
#         plt.imshow(my_targets[s])
#     else:
#         plt.subplot(3, 6, j + 8)
#         plt.imshow(my_images[s])
#         plt.subplot(3, 6, j + 11)
#         plt.imshow(my_targets[s])
# plt.show()
# exit()