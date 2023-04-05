# from ComputationalGraphPrimer import *
import random
import numpy as np
import torch
import torch.nn as nn 
import math

"""
Loss logic for YOLO detection
"""

EPS = 1e-6

class YOLOLoss(nn.Module):
    """
    Loss for YOLO network, implemented based on the original paper.
    Input parameters:
        split_grids (int): number of cells in each direction (S)
        num_bboxes (int): number of bounding boxes to predict per cell
        num_classes (int): number of dataset classes
        lambda_coord (int): weight for location loss
        lambda_noobj (int): weight for the case where there is no object in the cell.
    """
    def __init__(self, split_grids=8, num_anchor=5, num_classes=3, lambda_coord=5, lambda_noobj=0.5) -> None:
        super().__init__()
        self.S = split_grids
        self.A = num_anchor
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.criterion1 = nn.BCELoss() # for the objectness score, applied to all of the boxes
        self.criterion2 = nn.MSELoss() # for the position of the bboxes, applied only when there is an object in the box
        self.criterion3 = nn.CrossEntropyLoss() # for the classification error, applied only if there is an object

    def forward(self, pred, target):
        """
        The function takes a prediction tensor of shape (batch_size, A, S, S, 5 + C)
        and target tensor of shape (batch_size, A, S, S, 5 + C)
        The last dimension is organized as [class probabilities, (score, x,y,w,h)]
        The output is a single float number -- resulting loss
        """

        # print("dtype pred = ", pred.dtype)
        # print("dtype pred = ", target.dtype)



        # Find indices where there is an object:
        Iobj_i = target[..., 0].bool()
        # print("Iobj_i.size() = ", Iobj_i.size())
        object_present_target = target[Iobj_i]
        # print("object_present_target.size() = ", object_present_target.size())
        object_present_pred = pred[Iobj_i]
        # print("object_present_pred.size() = ", object_present_pred.size())

        pred_bce = nn.Sigmoid()(pred[..., 0].unsqueeze(-1))
        # print("pred BCEEEEE", torch.max(pred_bce), torch.min(pred_bce))
        # print("pred BCEEEEE", torch.max(target[..., 0].unsqueeze(-1)), torch.min(target[..., 0].unsqueeze(-1)))

        loss1 = self.criterion1(pred_bce, target[..., 0].unsqueeze(-1)) \
            + 10 * self.criterion1(nn.Sigmoid()(object_present_pred[..., 0].unsqueeze(-1)), object_present_target[..., 0].unsqueeze(-1))# Separately for objects
        loss2 = self.criterion2(object_present_pred[..., 1:5], object_present_target[..., 1:5])
        loss3 = self.criterion3(object_present_pred[..., 5:], object_present_target[..., 5:])
        # print(loss1, loss2, loss3)
        # print(object_present_target, object_present_pred)

        return loss1, loss2, loss3

if __name__=="__main__":
    pred = torch.rand(16, 5, 8, 8, 23)
    target = torch.rand(16, 5, 8, 8, 23)
    loss = YOLOLoss()
    seed= 100
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    print(loss(pred, target))
