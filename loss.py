import torch
import torch.nn.functional as F

"""
Loss functions
"""

def dice_loss(logits, labels):
    mask = labels != -1
    probs = torch.sigmoid(logits)
    probs = probs * mask
    labels = (labels * mask).float()
    intersection = (probs * labels).sum()
    union = probs.sum() + labels.sum()
    return 1 - (2 * intersection + 1e-6) / (union + 1e-6)

def iou_loss(logits, labels):
    mask = labels != -1
    probs = torch.sigmoid(logits)
    probs = probs * mask
    labels = (labels * mask).float()
    intersection = (probs * labels).sum()
    union = (probs + labels).sum() - intersection
    return 1 - (intersection + 1e-6) / (union + 1e-6)

def bce_loss(logits, labels):
    mask = labels != -1
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-6)