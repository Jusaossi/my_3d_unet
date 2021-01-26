import torch


def calculate_my_sets(inputs, targets, smooth=1e-5):
    # flatten label and prediction tensors
    inputs = torch.reshape(inputs, (-1,))
    targets = torch.reshape(targets, (-1,))
    # True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()
    return TP.item(), FP.item(), FN.item()


def calculate_my_metrics(TP, FP, FN, smooth=1e-5):
    recall = (TP + smooth) / (TP + FN + smooth)
    #true_negative_rate = (TN + smooth) / (TN + FP + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_score = (2 * precision * recall + smooth) / (precision + recall + smooth)
    return recall, precision, f1_score
