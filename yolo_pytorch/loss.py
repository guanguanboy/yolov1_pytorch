import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lamdba_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) #Iobj_i

        #for box coordinates
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30] #如果第2个boundingbox是bestbox，bestbox值为1
                + (1 - bestbox) * predictions[..., 21:25] #如果第1个bounding是bestbox，bestbox值为0
            )
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6)) # box的width，height

        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        #(N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        
        #for object loss
        pred_box = (
            bestbox * predictions[..., 25:26] #第2个box是bestbox，bestbox值为1 
            + (1 - bestbox) * predictions[..., 20:21] #第一个box是bestbox，bestbox值为0
        ) #这一步可以查找到是哪一个box复杂检测这个物体

        #（N * S * S）
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # for no object loss
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        ) #惩罚第1个box中不包含object

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        ) #惩罚第2个box中不包含object

        # for class loss
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = (
            self.lamdba_coord * box_loss # First two rows of loss in paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss