import numpy as np
import os
import matplotlib.pyplot as plt
import platform

machine = platform.node()

for j in range(1, 8):
    batch_nro = j + 231 # 196
    print('batsi nro=', batch_nro)
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
                  190: ['S0157', 'batch_1'], 191: ['S0157', 'batch_2'], 192: ['S0157', 'batch_3'],
                  193: ['S0157', 'batch_4'],
                  194: ['S0157', 'batch_5'], 195: ['S0157', 'batch_6'], 196: ['S0157', 'batch_7'],
                  197: ['S0157', 'batch_8'],
                  198: ['S0157', 'batch_9'], 199: ['S0157', 'batch_10'], 200: ['S0157', 'batch_11'],
                  201: ['S0157', 'batch_12'],
                  202: ['S0157', 'batch_13'], 203: ['S0157', 'batch_14'], 204: ['S0157', 'batch_15'],
                  205: ['S0157', 'batch_16'],
                  206: ['S0157', 'batch_17'], 207: ['S0157', 'batch_18'], 208: ['S0157', 'batch_19'],
                  209: ['S0157', 'batch_20'],
                  210: ['S0157', 'batch_21'],
                  211: ['S0406_1', 'batch_1'], 212: ['S0406_1', 'batch_2'], 213: ['S0406_1', 'batch_3'],
                  214: ['S0406_1', 'batch_4'],
                  215: ['S0406_1', 'batch_5'], 216: ['S0406_1', 'batch_6'], 217: ['S0406_1', 'batch_7'],
                  218: ['S0406_1', 'batch_8'],
                  219: ['S0406_1', 'batch_9'], 220: ['S0406_1', 'batch_10'], 221: ['S0406_1', 'batch_11'],
                  222: ['S0406_1', 'batch_12'],
                  223: ['S0406_1', 'batch_13'], 224: ['S0406_1', 'batch_14'], 225: ['S0406_1', 'batch_15'],
                  226: ['S0406_1', 'batch_16'],
                  227: ['S0406_1', 'batch_17'], 228: ['S0406_1', 'batch_18'], 229: ['S0406_1', 'batch_19'],
                  230: ['S0406_1', 'batch_20'],
                  231: ['S0406_1', 'batch_21'],
                  232: ['visoerkki', 'batch_1'], 233: ['visoerkki', 'batch_2'], 234: ['visoerkki', 'batch_3'],
                  235: ['visoerkki', 'batch_4'],
                  236: ['visoerkki', 'batch_5'], 237: ['visoerkki', 'batch_6'], 238: ['visoerkki', 'batch_7'],
                  239: ['visoerkki', 'batch_8'],
                  240: ['visoerkki', 'batch_9'], 241: ['visoerkki', 'batch_10'], 242: ['visoerkki', 'batch_11'],
                  243: ['visoerkki', 'batch_12'],
                  244: ['visoerkki', 'batch_13'], 245: ['visoerkki', 'batch_14'], 246: ['visoerkki', 'batch_15'],
                  247: ['visoerkki', 'batch_16'],
                  248: ['visoerkki', 'batch_17'], 249: ['visoerkki', 'batch_18'], 250: ['visoerkki', 'batch_19'],
                  251: ['visoerkki', 'batch_20'],
                  252: ['visoerkki', 'batch_21']
                  }

    patient = my_batches[batch_nro][0]

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
        add_x = 10
        add_x2 = 5
        cube_size = 96
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
        cube_size = 96
        add_x = 20
        add_x2 = 10
    elif patient == 'timo':
        batch_cut = [440, 600, 80, 230, 240, 440]
        x_half_length = int(200 / 2)  # 66,66 minimi
        y_half_length = int(150 / 2)  # 63,33
        z_half_length = int(160 / 2)
        cube_size = 96
        add_x = 20
        add_x2 = 10

    elif patient == 'patient1':
        batch_cut = [40, 420, 26, 410, 30, 520]
        my_move = 80
        my_move_2 = 60
        x_half_length = int((514 - 30) / 2)  # 66,66 minimi
        x_quarter_length = int((514 - 30) / 4)
        y_half_length = int((410 - 26) / 2)  # 63,33
        y_quarter_length = int((410 - 26) / 4)  # 63,33
        z_half_length = int(380 / 2)
        z_quarter_length = int(380 / 4)
        cube_size = 96
        cube_size_z = 96
        print(y_quarter_length, y_half_length)

    elif patient == 'S0157':
        batch_cut = [0, 150, 0, 160, 0, 180]
        x_half_length = int(180 / 2)  # 66,66 minimi
        y_half_length = int(160 / 2)  # 63,33
        z_half_length = int(150 / 2)
        cube_size = 96
        add_x = 10
        add_x2 = 5

    elif patient == 'S0406_1':
        batch_cut = [0, 240, 0, 240, 0, 300]
        x_half_length = int(300 / 2)  # 66,66 minimi
        y_half_length = int(240 / 2)  # 63,33
        z_half_length = int(240 / 2)
        cube_size = 96
        add_x = 50
        add_x2 = 25

    elif patient == 'visoerkki':
        batch_cut = [0, 130, 0, 190, 0, 240]
        x_half_length = int(240 / 2)  # 66,66 minimi
        y_half_length = int(190 / 2)  # 63,33
        z_half_length = int(130 / 2)
        cube_size = 96
        add_x = 10
        add_x2 = 5
    # print(patient, cube_size)
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
                     batch_cut[5] - x_quarter_length- int(my_move_2 / 2) + int(cube_size / 2)],
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
    my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = batch_coordinates2[my_batches[batch_nro + 0][1]]
    print(my_slicer1)
    print(patient)
    if patient != 'patient1':
        my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = batch_coordinates[
            my_batches[batch_nro][1]]

    else:
        # print('mulkku')
        my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6 = batch_coordinates2[
            my_batches[batch_nro][1]]
    print(my_slicer1, my_slicer2, my_slicer3, my_slicer4, my_slicer5, my_slicer6)
    if machine == 'DESKTOP-K3R0DFP':
        my_path = r'C:\Users\jpkorpel\Desktop\hammas'
    else:
        my_dir = os.getcwd()
        my_path = os.path.join(my_dir, 'hammas')
    # patient = 'patient1'
    patient_images_file = 'X_images_' + patient + '.npy'
    patient_targets_file = 'Y_targets_' + patient + '.npy'
    image_folder = os.path.join(my_path, patient, patient_images_file)
    # print(image_folder)
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



    print(my_images.shape)
    print(my_targets.shape)
    s = 30
    print('my j', j)

    if j in [1, 2, 3, 4, 5]:
        plt.subplot(5, 10, j)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 5)
        plt.imshow(my_targets[s])
    elif j in [6, 7]:
        plt.subplot(5, 10, j + 5)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 10)
        plt.imshow(my_targets[s])
    elif j in [8, 9]:
        plt.subplot(5, 10, j + 6)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 11)
        plt.imshow(my_targets[s])
    elif j in [10, 11]:
        plt.subplot(5, 10, j + 11)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 16)
        plt.imshow(my_targets[s])
    elif j in [12, 13]:
        plt.subplot(5, 10, j + 12)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 17)
        plt.imshow(my_targets[s])
    elif j in [14, 15]:
        plt.subplot(5, 10, j + 17)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 22)
        plt.imshow(my_targets[s])
    elif j in [16, 17]:
        plt.subplot(5, 10, j + 18)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 23)
        plt.imshow(my_targets[s])
    elif j in [18, 19]:
        plt.subplot(5, 10, j + 23)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 28)
        plt.imshow(my_targets[s])
    elif j in [20, 21]:
        plt.subplot(5, 10, j + 24)
        plt.imshow(my_images[s])
        plt.subplot(5, 10, j + 29)
        plt.imshow(my_targets[s])
plt.show()
exit()
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