import torch
from torch import nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def __init__(self, label_smoothing=0):
        super(ClipLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits_per_image, logits_per_text, labels=None):  
        if not labels:
            # logits_per_image, logits_per_text, labels are all the same size (batch, batch)
            labels = torch.arange(logits_per_image.size(0)).long().cuda()
            
        loss_i = self.loss_func(logits_per_image, labels)
        loss_t = self.loss_func(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss

class ClipHardNegSampleLoss(nn.Module):
    def __init__(self, label_smoothing=0):
        super(ClipHardNegSampleLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits_per_image, labels=None):  
        if not labels:
            labels = torch.arange(logits_per_image.size(0)).long().cuda() 
        loss = self.loss_func(logits_per_image, labels)
        return loss
    
class TopNLabelSmoothingCrossEntropy(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, class_num, topN_target=[0.7, 0.2, 0.1]):
        super().__init__()
        assert len(topN_target) <= class_num, 'length of topN_target should be smaller than class_num'
        self.class_num = class_num
        self.topN_target = topN_target

    def forward(self, preds, targets):
        logprobs = F.log_softmax(preds, dim=1)  # softmax + log
        targets = F.one_hot(targets, self.class_num).type_as(logprobs)  # convert to one-hot

        for i in range(self.class_num):
            logprob, target = logprobs[i, :], targets[i, :]  # [class_num]
            _, sorted_indices = torch.sort(logprob, dim=0, descending=True)

            gt_idx = torch.nonzero(sorted_indices == i).item()
            tmp_idx = 0
            for j in range(len(self.topN_target)):
                if j == 0:
                    update_idx = sorted_indices[gt_idx].item()  # update gt target prob
                else:
                    if tmp_idx == gt_idx:
                        tmp_idx += 1
                    update_idx = sorted_indices[tmp_idx].item()
                    tmp_idx += 1
                target[update_idx] = self.topN_target[j]
            targets[i, :] = target
        loss = -1 * torch.sum(targets * logprobs, 1)
        return loss.mean()

class TopNLabelSmoothingClipLoss(nn.Module):
    def __init__(self, class_num, topN_target=[0.7, 0.2, 0.1]):
        super(TopNLabelSmoothingClipLoss, self).__init__()
        self.loss_func = TopNLabelSmoothingCrossEntropy(class_num=class_num, topN_target=topN_target)

    def forward(self, logits_per_image, logits_per_text, labels=None):  
        if not labels:
            # logits_per_image, logits_per_text, labels are all the same size (batch, batch)
            labels = torch.arange(logits_per_image.size(0)).long().cuda()
            
        loss_i = self.loss_func(logits_per_image, labels)
        loss_t = self.loss_func(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss
    
class SkipTopNCrossEntropy(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, class_num, label_smooth=0.1, skip_num=3):
        super().__init__()
        assert skip_num <= class_num, 'length of topN_target should be smaller than class_num'
        self.class_num = class_num
        self.label_smooth = label_smooth
        self.skip_num = skip_num

    def forward(self, preds, targets):
        logprobs = F.log_softmax(preds, dim=1)  # softmax + log
        targets = F.one_hot(targets, self.class_num)  # convert to one-hot
        targets = torch.clamp(targets.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)

        loss = torch.zeros([self.class_num, 1]).cuda()
        for i in range(self.class_num):
            logprob, target = logprobs[i, :], targets[i, :]  # [class_num]
            _, sorted_indices = torch.sort(logprob, dim=0, descending=True)

            gt_idx = torch.nonzero(sorted_indices == i).item()
            tmp_idx = 0
            skip_indices = []
            for _ in range(self.skip_num):
                if tmp_idx == gt_idx:
                    tmp_idx += 1
                update_idx = sorted_indices[tmp_idx].item()
                skip_indices.append(update_idx)
                tmp_idx += 1
            new_logprob = logprob[[idx for idx in range(logprob.size(0)) if idx not in skip_indices]]
            new_target = target[[idx for idx in range(target.size(0)) if idx not in skip_indices]]
            loss[i] = -torch.sum(new_target * new_logprob)
        return loss.mean()

class SkipTopNClipLoss(nn.Module):
    def __init__(self, class_num, label_smooth=0.1, skip_num=2):
        super(SkipTopNClipLoss, self).__init__()
        self.loss_func = SkipTopNCrossEntropy(class_num=class_num, label_smooth=label_smooth, skip_num=skip_num)

    def forward(self, logits_per_image, logits_per_text, labels=None):  
        if not labels:
            # logits_per_image, logits_per_text, labels are all the same size (batch, batch)
            labels = torch.arange(logits_per_image.size(0)).long().cuda()
            
        loss_i = self.loss_func(logits_per_image, labels)
        loss_t = self.loss_func(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss
