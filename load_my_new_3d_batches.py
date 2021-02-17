import numpy as np
import os
import matplotlib.pyplot as plt
import platform
import random
from my_3d_albumentations import my_3d_random_crop

machine = platform.node()


def load_my_new_3d_batch(batch_nro, scale, lower_cut, crop_cube, crop_strategy):
    my_batches = {1: ['andy', 'batch_1'], 2: ['andy', 'batch_2'], 3: ['andy', 'batch_3'], 4: ['andy', 'batch_4'],
                  5: ['andy', 'batch_5'], 6: ['andy', 'batch_6'], 7: ['andy', 'batch_7'], 8: ['andy', 'batch_8'],
                  9: ['andy', 'batch_9'], 10: ['andy', 'batch_10'], 11: ['andy', 'batch_11'],
                  12: ['andy', 'batch_12'],
                  13: ['andy', 'batch_13'], 14: ['andy', 'batch_14'], 15: ['andy', 'batch_15'],
                  16: ['andy', 'batch_16'],
                  17: ['andy', 'batch_17'], 18: ['andy', 'batch_18'], 19: ['andy', 'batch_19'],
                  20: ['andy', 'batch_20'],
                  21: ['andy', 'batch_21'],
                  22: ['teeth1', 'batch_1'], 23: ['teeth1', 'batch_2'], 24: ['teeth1', 'batch_3'],
                  25: ['teeth1', 'batch_4'],
                  26: ['teeth1', 'batch_5'], 27: ['teeth1', 'batch_6'], 28: ['teeth1', 'batch_7'],
                  29: ['teeth1', 'batch_8'],
                  30: ['teeth1', 'batch_9'], 31: ['teeth1', 'batch_10'], 32: ['teeth1', 'batch_11'],
                  33: ['teeth1', 'batch_12'],
                  34: ['teeth1', 'batch_13'], 35: ['teeth1', 'batch_14'], 36: ['teeth1', 'batch_15'],
                  37: ['teeth1', 'batch_16'],
                  38: ['teeth1', 'batch_17'], 39: ['teeth1', 'batch_18'], 40: ['teeth1', 'batch_19'],
                  41: ['teeth1', 'batch_20'],
                  42: ['teeth1', 'batch_21'],
                  43: ['teeth2', 'batch_1'], 44: ['teeth2', 'batch_2'], 45: ['teeth2', 'batch_3'],
                  46: ['teeth2', 'batch_4'],
                  47: ['teeth2', 'batch_5'], 48: ['teeth2', 'batch_6'], 49: ['teeth2', 'batch_7'],
                  50: ['teeth2', 'batch_8'],
                  51: ['teeth2', 'batch_9'], 52: ['teeth2', 'batch_10'], 53: ['teeth2', 'batch_11'],
                  54: ['teeth2', 'batch_12'],
                  55: ['teeth2', 'batch_13'], 56: ['teeth2', 'batch_14'], 57: ['teeth2', 'batch_15'],
                  58: ['teeth2', 'batch_16'],
                  59: ['teeth2', 'batch_17'], 60: ['teeth2', 'batch_18'], 61: ['teeth2', 'batch_19'],
                  62: ['teeth2', 'batch_20'],
                  63: ['teeth2', 'batch_21'],
                  64: ['patient1', 'batch_1'], 65: ['patient1', 'batch_2'], 66: ['patient1', 'batch_3'],
                  67: ['patient1', 'batch_4'], 68: ['patient1', 'batch_5'], 69: ['patient1', 'batch_6'],
                  70: ['patient1', 'batch_7'],
                  71: ['patient1', 'batch_8'], 72: ['patient1', 'batch_9'], 73: ['patient1', 'batch_10'],
                  74: ['patient1', 'batch_11'],
                  75: ['patient1', 'batch_12'], 76: ['patient1', 'batch_13'], 77: ['patient1', 'batch_14'],
                  78: ['patient1', 'batch_15'],
                  79: ['patient1', 'batch_16'], 80: ['patient1', 'batch_17'], 81: ['patient1', 'batch_18'],
                  82: ['patient1', 'batch_19'],
                  83: ['patient1', 'batch_20'], 84: ['patient1', 'batch_21'], 85: ['patient1', 'batch_22'],
                  86: ['patient1', 'batch_23'],
                  87: ['patient1', 'batch_24'], 88: ['patient1', 'batch_25'], 89: ['patient1', 'batch_26'],
                  90: ['patient1', 'batch_27'],
                  91: ['patient1', 'batch_28'], 92: ['patient1', 'batch_29'], 93: ['patient1', 'batch_30'],
                  94: ['patient1', 'batch_31'],
                  95: ['patient1', 'batch_32'], 96: ['patient1', 'batch_33'], 97: ['patient1', 'batch_34'],
                  98: ['patient1', 'batch_35'], 99: ['patient1', 'batch_36'], 100: ['patient1', 'batch_37'],
                  101: ['patient1', 'batch_38'], 102: ['patient1', 'batch_39'], 103: ['patient1', 'batch_40'],
                  104: ['patient1', 'batch_41'],
                  105: ['patient1', 'batch_42'], 106: ['patient1', 'batch_43'], 107: ['patient1', 'batch_44'],
                  108: ['patient1', 'batch_45'],
                  109: ['patient1', 'batch_46'], 110: ['patient1', 'batch_47'], 111: ['patient1', 'batch_48'],
                  112: ['patient1', 'batch_49'],
                  113: ['patient1', 'batch_50'], 114: ['patient1', 'batch_51'], 115: ['patient1', 'batch_52'],
                  116: ['patient1', 'batch_53'],
                  117: ['patient1', 'batch_54'], 118: ['patient1', 'batch_55'], 119: ['patient1', 'batch_56'],
                  120: ['patient1', 'batch_57'],
                  121: ['patient1', 'batch_58'], 122: ['patient1', 'batch_59'], 123: ['patient1', 'batch_60'],
                  124: ['patient1', 'batch_61'],
                  125: ['patient1', 'batch_62'], 126: ['patient1', 'batch_63'], 127: ['patient1', 'batch_64'],
                  128: ['patient1', 'batch_65'],
                  129: ['patient1', 'batch_66'],
                  130: ['patient1', 'batch_67'], 131: ['patient1', 'batch_68'], 132: ['patient1', 'batch_69'],
                  133: ['patient1', 'batch_70'],
                  134: ['patient1', 'batch_71'], 135: ['patient1', 'batch_72'], 136: ['patient1', 'batch_73'],
                  137: ['patient1', 'batch_74'],
                  138: ['patient1', 'batch_75'], 139: ['patient1', 'batch_76'], 140: ['patient1', 'batch_77'],
                  141: ['patient1', 'batch_78'],
                  142: ['patient1', 'batch_79'],
                  143: ['patient1', 'batch_80'], 144: ['patient1', 'batch_81'], 145: ['patient1', 'batch_82'],
                  146: ['patient1', 'batch_83'],
                  147: ['patient1', 'batch_84'], 148: ['patient1', 'batch_85'], 149: ['patient1', 'batch_86'],
                  150: ['patient1', 'batch_87'],
                  151: ['patient1', 'batch_88'], 152: ['patient1', 'batch_89'], 153: ['patient1', 'batch_90'],
                  154: ['patient1', 'batch_91'],
                  155: ['patient1', 'batch_92'], 156: ['patient1', 'batch_93'], 157: ['patient1', 'batch_94'],
                  158: ['patient1', 'batch_95'],
                  159: ['patient1', 'batch_96'], 160: ['patient1', 'batch_97'], 161: ['patient1', 'batch_98'],
                  162: ['patient1', 'batch_99'], 163: ['patient1', 'batch_100'], 164: ['patient1', 'batch_101'],
                  165: ['patient1', 'batch_102'], 166: ['patient1', 'batch_103'], 167: ['patient1', 'batch_104'],
                  168: ['patient1', 'batch_105'],
                  169: ['timo', 'batch_1'], 170: ['timo', 'batch_2'], 171: ['timo', 'batch_3'],
                  172: ['timo', 'batch_4'],
                  173: ['timo', 'batch_5'], 174: ['timo', 'batch_6'], 175: ['timo', 'batch_7'],
                  176: ['timo', 'batch_8'],
                  177: ['timo', 'batch_9'], 178: ['timo', 'batch_10'], 179: ['timo', 'batch_11'],
                  180: ['timo', 'batch_12'],
                  181: ['timo', 'batch_13'], 182: ['timo', 'batch_14'], 183: ['timo', 'batch_15'],
                  184: ['timo', 'batch_16'],
                  185: ['timo', 'batch_17'], 186: ['timo', 'batch_18'], 187: ['timo', 'batch_19'],
                  188: ['timo', 'batch_20'],
                  189: ['timo', 'batch_21'],
                  }

    patient = my_batches[batch_nro][0]
    # print(patient)
    batch_cut = None
    x_half_length = None
    y_half_length = None
    z_half_length = None
    z_quarter_length = 1
    cube_size = None
    cube_size_z = 1
    my_move = 0
    my_move_2 = 0
    x_quarter_length = 0
    y_quarter_length = 0

    add_x = 0
    add_x2 = 0
    if patient == 'andy':
        batch_cut = [470, 620, 120, 250, 250, 410]
        x_half_length = int(160 / 2)
        y_half_length = int(130 / 2)
        z_half_length = int(150 / 2)
        add_x = 20
        add_x2 = 10
        cube_size = 80
    elif patient == 'teeth1':
        batch_cut = [30, 194, 50, 280, 100, 350]
        x_half_length = int(250 / 2)  # 70 mimimi
        y_half_length = int(230 / 2)  # 77 minimi
        z_half_length = int(164 / 2)
        cube_size = 96
        add_x = 20
        add_x2 = 10
    elif patient == 'teeth2':
        batch_cut = [150, 306, 20, 210, 150, 390]
        x_half_length = int(240 / 2)  # 66,66 minimi
        y_half_length = int(190 / 2)  # 63,33
        z_half_length = int(156 / 2)
        cube_size = 88
        add_x = 20
        add_x2 = 10
    elif patient == 'timo':
        batch_cut = [440, 600, 80, 230, 240, 440]
        x_half_length = int(200 / 2)  # 66,66 minimi
        y_half_length = int(150 / 2)  # 63,33
        z_half_length = int(160 / 2)
        cube_size = 80
        add_x = 20
        add_x2 = 10

    elif patient == 'patient1':
        batch_cut = [40, 420, 26, 410, 30, 520]
        my_move = 80
        my_move_2 = 40
        x_half_length = int((514 - 30) / 2)  # 66,66 minimi
        x_quarter_length = int((514 - 30) / 4)
        y_half_length = int((410 - 26) / 2)  # 63,33
        y_quarter_length = int((410 - 26) / 4)  # 63,33
        z_half_length = int(380 / 2)
        z_quarter_length = int(380 / 4)
        cube_size = 96
        cube_size_z = 96

    #print(patient, cube_size)
    # print(my_batches[batch_nro][1])

    # batch_coordinates ={'batch_1': [batch_cut[0], batch_cut[0] + cube_size, batch_cut[2], batch_cut[2] + cube_size, batch_cut[4] + 20, batch_cut[4] + 20 + cube_size]}


    # my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = my_batches[batch_nro][1]


    # andy_cut = [470, 620, 120, 250, 250, 410]
    batch_coordinates = {
        'batch_1': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + add_x, batch_cut[4] + add_x + cube_size],
        'batch_2': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_half_length - int(cube_size / 2),
                    batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_3': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[5] - add_x - cube_size, batch_cut[5] - add_x],
        'batch_4': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2] + y_half_length - int(cube_size / 2),
                    batch_cut[2] + y_half_length + int(cube_size / 2),
                    batch_cut[4] + add_x2, batch_cut[4] + add_x2 + cube_size],
        'batch_5': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[2] + y_half_length - int(cube_size / 2),
                    batch_cut[2] + y_half_length + int(cube_size / 2),
                    batch_cut[5] - add_x2 - cube_size, batch_cut[5] - add_x2],
        'batch_6': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[3] - cube_size, batch_cut[3],
                    batch_cut[4], batch_cut[4] + cube_size],
        'batch_7': [batch_cut[0], batch_cut[0] + cube_size,
                    batch_cut[3] - cube_size, batch_cut[3],
                    batch_cut[5] - cube_size, batch_cut[5]],
        'batch_8': [batch_cut[0] + z_half_length - int(cube_size / 2),
                    batch_cut[0] + z_half_length + int(cube_size / 2),
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + add_x, batch_cut[4] + add_x + cube_size],
        'batch_9': [batch_cut[0] + z_half_length - int(cube_size / 2),
                    batch_cut[0] + z_half_length + int(cube_size / 2),
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_half_length - int(cube_size / 2),
                    batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_10': [batch_cut[0] + z_half_length - int(cube_size / 2),
                     batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - add_x - cube_size, batch_cut[5] - add_x],
        'batch_11': [batch_cut[0] + z_half_length - int(cube_size / 2),
                     batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + add_x2, batch_cut[4] + add_x2 + cube_size],
        'batch_12': [batch_cut[0] + z_half_length - int(cube_size / 2),
                     batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - add_x2 - cube_size, batch_cut[5] - add_x2],
        'batch_13': [batch_cut[0] + z_half_length - int(cube_size / 2),
                     batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_14': [batch_cut[0] + z_half_length - int(cube_size / 2),
                     batch_cut[0] + z_half_length + int(cube_size / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],
        'batch_15': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + add_x, batch_cut[4] + add_x + cube_size],
        'batch_16': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_half_length - int(cube_size / 2),
                     batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_17': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - add_x - cube_size, batch_cut[5] - add_x],
        'batch_18': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + add_x2, batch_cut[4] + add_x2 + cube_size],
        'batch_19': [batch_cut[1] - cube_size, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - add_x2 - cube_size, batch_cut[5] - add_x2],
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
                    batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_2': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                    batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_3': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[4] + x_half_length - int(cube_size / 2),
                    batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_4': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                    batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_5': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2], batch_cut[2] + cube_size,
                    batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_6': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2] + y_quarter_length - int(cube_size / 2),
                    batch_cut[2] + y_quarter_length + int(cube_size / 2),
                    batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_7': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2] + y_quarter_length - int(cube_size / 2),
                    batch_cut[2] + y_quarter_length + int(cube_size / 2),
                    batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                    batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_8': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2] + y_quarter_length - int(cube_size / 2),
                    batch_cut[2] + y_quarter_length + int(cube_size / 2),
                    batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                    batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_9': [batch_cut[0], batch_cut[0] + cube_size_z,
                    batch_cut[2] + y_quarter_length - int(cube_size / 2),
                    batch_cut[2] + y_quarter_length + int(cube_size / 2),
                    batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_10': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + my_move_2, batch_cut[4] + my_move_2 + cube_size],
        'batch_11': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_12': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_13': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - my_move_2 - cube_size, batch_cut[5] - my_move_2],

        'batch_14': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_15': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_16': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_17': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_18': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_19': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_20': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - x_quarter_length - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_21': [batch_cut[0], batch_cut[0] + cube_size_z,
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_22': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_23': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_24': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_half_length - int(cube_size / 2),
                     batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_25': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_26': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_27': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_28': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_29': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_30': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_31': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + my_move_2, batch_cut[4] + my_move_2 + cube_size],
        'batch_32': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_33': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_34': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - my_move_2 - cube_size, batch_cut[5] - my_move_2],

        'batch_35': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_36': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_37': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_38': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_39': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_40': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_41': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - x_quarter_length - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_42': [batch_cut[0] + z_quarter_length - int(cube_size_z / 2),
                     batch_cut[0] + z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_43': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_44': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_45': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_half_length - int(cube_size / 2),
                     batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_46': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_47': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_48': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_49': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_50': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_51': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_52': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + my_move_2, batch_cut[4] + my_move_2 + cube_size],
        'batch_53': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_54': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(my_move_2) - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(my_move_2) + int(cube_size / 2)],
        'batch_55': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_56': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_57': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_58': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_59': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_60': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_61': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_62': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - x_quarter_length - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_63': [batch_cut[0] + z_half_length - int(cube_size_z / 2),
                     batch_cut[0] + z_half_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_64': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_65': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],

        'batch_66': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_half_length - int(cube_size / 2),
                     batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_67': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_68': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_69': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_70': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_71': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_72': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_73': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + my_move_2, batch_cut[4] + my_move_2 + cube_size],
        'batch_74': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_75': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_76': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - my_move_2 - cube_size, batch_cut[5] - my_move_2],

        'batch_77': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_78': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_79': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_80': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_81': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_82': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_83': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - x_quarter_length - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_84': [batch_cut[1] - z_quarter_length - int(cube_size_z / 2),
                     batch_cut[1] - z_quarter_length + int(cube_size_z / 2),
                     batch_cut[3] - cube_size, batch_cut[3],
                     batch_cut[5] - cube_size, batch_cut[5]],

        'batch_85': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_86': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_87': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[4] + x_half_length - int(cube_size / 2),
                     batch_cut[4] + x_half_length + int(cube_size / 2)],
        'batch_88': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - x_quarter_length - my_move_2 - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - my_move_2 + int(cube_size / 2)],
        'batch_89': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2], batch_cut[2] + cube_size,
                     batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_90': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + my_move, batch_cut[4] + my_move + cube_size],
        'batch_91': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + my_move_2 + int(cube_size / 2)],
        'batch_92': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_93': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_quarter_length - int(cube_size / 2),
                     batch_cut[2] + y_quarter_length + int(cube_size / 2),
                     batch_cut[5] - my_move - cube_size, batch_cut[5] - my_move],

        'batch_94': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + my_move_2, batch_cut[4] + my_move_2 + cube_size],
        'batch_95': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_96': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) - int(cube_size / 2),
                     batch_cut[5] - x_quarter_length - int(my_move_2 / 2) + int(cube_size / 2)],
        'batch_97': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[2] + y_half_length - int(cube_size / 2),
                     batch_cut[2] + y_half_length + int(cube_size / 2),
                     batch_cut[5] - my_move_2 - cube_size, batch_cut[5] - my_move_2],

        'batch_98': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4], batch_cut[4] + cube_size],
        'batch_99': [batch_cut[1] - cube_size_z, batch_cut[1],
                     batch_cut[3] - y_quarter_length - int(cube_size / 2),
                     batch_cut[3] - y_quarter_length + int(cube_size / 2),
                     batch_cut[4] + x_quarter_length - int(cube_size / 2),
                     batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_100': [batch_cut[1] - cube_size_z, batch_cut[1],
                      batch_cut[3] - y_quarter_length - int(cube_size / 2),
                      batch_cut[3] - y_quarter_length + int(cube_size / 2),
                      batch_cut[5] - x_quarter_length - int(cube_size / 2),
                      batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_101': [batch_cut[1] - cube_size_z, batch_cut[1],
                      batch_cut[3] - y_quarter_length - int(cube_size / 2),
                      batch_cut[3] - y_quarter_length + int(cube_size / 2),
                      batch_cut[5] - cube_size, batch_cut[5]],

        'batch_102': [batch_cut[1] - cube_size_z, batch_cut[1],
                      batch_cut[3] - cube_size, batch_cut[3],
                      batch_cut[4], batch_cut[4] + cube_size],
        'batch_103': [batch_cut[1] - cube_size_z, batch_cut[1],
                      batch_cut[3] - cube_size, batch_cut[3],
                      batch_cut[4] + x_quarter_length - int(cube_size / 2),
                      batch_cut[4] + x_quarter_length + int(cube_size / 2)],
        'batch_104': [batch_cut[1] - cube_size_z, batch_cut[1],
                      batch_cut[3] - cube_size, batch_cut[3],
                      batch_cut[5] - x_quarter_length - int(cube_size / 2),
                      batch_cut[5] - x_quarter_length + int(cube_size / 2)],
        'batch_105': [batch_cut[1] - cube_size_z, batch_cut[1],
                      batch_cut[3] - cube_size, batch_cut[3],
                      batch_cut[5] - cube_size, batch_cut[5]],
    }
    # print(batch_coordinates[my_batches[batch_nro][1]])
    #my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = batch_coordinates[my_batches[batch_nro + 0][1]]
    # print(my_slicer1)
    if patient != 'patient1':
        my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = batch_coordinates[my_batches[batch_nro][1]]
        
    else:
        # print('mulkku')
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
    my_crop = random.choices(['True', 'False'], weights=(30, 70), k=1)[0]
    # print('is crop=', my_crop)
    if crop_cube != 'False' and my_crop == 'True':
        if crop_strategy == 'random_small':
            crop_slicer1, crop_slicer2, crop_slicer3, crop_slicer4, crop_slicer5, crop_slicer6 = my_3d_random_crop(my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6, crop_cube)
            my_images = np.load(image_folder)[crop_slicer1: crop_slicer2, crop_slicer3: crop_slicer4, crop_slicer5: crop_slicer6]
            my_targets = np.load(target_folder)[crop_slicer1: crop_slicer2, crop_slicer3: crop_slicer4, crop_slicer5: crop_slicer6]
            my_images = my_images.astype(np.float32)
        elif crop_strategy == 'random_same':
            crop_slicer1, crop_slicer2, crop_slicer3, crop_slicer4, crop_slicer5, crop_slicer6 = my_3d_random_crop(batch_cut[0], batch_cut[1], batch_cut[2], batch_cut[3], batch_cut[4], batch_cut[5], cube_size)
            my_images = np.load(image_folder)[crop_slicer1: crop_slicer2, crop_slicer3: crop_slicer4, crop_slicer5: crop_slicer6]
            my_targets = np.load(target_folder)[crop_slicer1: crop_slicer2, crop_slicer3: crop_slicer4, crop_slicer5: crop_slicer6]
            my_images = my_images.astype(np.float32)
        # print('shape of my images =', my_images.shape)
        # print('crop_strategy=', crop_strategy)
        # print('cube size=', cube_size)
        # print('my_sliceri=', my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6)
        # print('my cube=', crop_slicer1, crop_slicer2, crop_slicer3, crop_slicer4, crop_slicer5, crop_slicer6)
        # print('my dicom=', batch_cut)

    else:
        # print('no_crop')
        my_images = np.load(image_folder)[my_slicer1: my_slicer2, my_slicer3: my_slicer4, my_slicer5: my_slicer6]
        my_targets = np.load(target_folder)[my_slicer1: my_slicer2, my_slicer3: my_slicer4, my_slicer5: my_slicer6]
        my_images = my_images.astype(np.float32)
        # print('shape of my images =', my_images.shape)

    # print(my_images.shape)
    # print('max =', np.max(my_images))
    # print('min =', np.min(my_images))
    # print(scale)
    if scale == '[-1,1]':
        np.clip(my_images, 500, 3500, out=my_images)
        my_images = my_images - 2000
        my_images = my_images / 1500
    elif scale == '[0,1]':
        if patient == 'andy':
            my_images = my_images + 94
        elif patient == 'teeth1':
            my_images = my_images - 162
        elif patient == 'teeth2':
            my_images = my_images + 41
        elif patient == 'patient1':
            my_images = my_images - 115
        elif patient == 'timo':
            my_images = my_images + 132
        np.clip(my_images, lower_cut, 4000, out=my_images)
        my_images = my_images - lower_cut
        my_images = my_images / (4000 - lower_cut)

    elif scale == 'norm':
        my_std = np.std(my_images)
        my_mean = np.mean(my_images)
        my_images = (my_images - my_mean) / my_std
    # print('after scaking', my_images.shape)
    # print('after scaling max =', np.max(my_images))
    # print('after scaling min =', np.min(my_images))
    # exit()
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