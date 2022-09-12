import glob
import numpy as np
import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable
import argparse
import os
 
pred_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='the path of dataset')
    parser.add_argument('--model', type=str, help='the path of model')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model)

    
    tests_path = os.listdir(args.dataset)
    os.makedirs('seg/',exist_ok=True)
    
    for test_path in tests_path:
        
        save_res_path = f'seg/{test_path}'     
        img = cv2.imread(f"{args.dataset}/{test_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256,256))
        img = pred_transform(img)
       
        inputs = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()
        pred = model(inputs)
        
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        
        pred = np.array(pred.data.cpu()[0])[0]

        cv2.imwrite(save_res_path, pred)