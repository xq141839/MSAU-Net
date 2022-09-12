import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from model import data_loading
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

def get_train_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='the path of dataset')
    parser.add_argument('--model', type=str, help='the path of model')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEST_PATH = args.dataset
    train_dataset = data_loading.medical_img_data(TEST_PATH, transform=get_train_transform())
    model = torch.load(args.model)
    acc_eval = Accuracy()
    pre_eval = Precision()
    dice_eval = Dice()
    recall_eval = Recall()
    f1_eval = F1(2)
    iou_eval = IoU()
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    
    since = time.time()
    
    with torch.no_grad():
        for img, mask in train_dataset:
            
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()           
            mask = Variable(mask.cuda())
            pred = model(img)
            
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            
            iouscore = iou_eval(pred,mask)
            dicescore = dice_eval(pred,mask)
            pred = pred.view(-1)
            mask = mask.view(-1)
            accscore = acc_eval(pred.cpu(),mask.cpu())
            prescore = pre_eval(pred.cpu(),mask.cpu())
            recallscore = recall_eval(pred.cpu(),mask.cpu())
            f1score = f1_eval(pred.cpu(),mask.cpu())
            iou_score.append(iouscore.cpu().detach().numpy())
            dice_score.append(dicescore.cpu().detach().numpy())
            acc_score.append(accscore.cpu().detach().numpy())
            pre_score.append(prescore.cpu().detach().numpy())
            recall_score.append(recallscore.cpu().detach().numpy())
            f1_score.append(f1score.cpu().detach().numpy())
            torch.cuda.empty_cache()
            
    time_elapsed = time.time() - since
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('mean IoU:',np.mean(iou_score),np.std(iou_score))
    print('mean accuracy:',np.mean(acc_score),np.std(acc_score))
    print('mean precsion:',np.mean(pre_score),np.std(pre_score))
    print('mean recall:',np.mean(recall_score),np.std(recall_score))
    print('mean F1-score:',np.mean(f1_score),np.std(f1_score))
        
