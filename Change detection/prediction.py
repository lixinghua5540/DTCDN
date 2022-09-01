import numpy as np
import os
import cv2
import torch
from torch import nn
from torch.optim import SGD,Adam
import torch.optim as optim
from  PIL import Image
from utils.Ranger import Ranger
import torch.nn.functional as F
from  siamunet_conc import SiamUnet_conc
from  siamunet_diff import SiamUnet_diff
#from Models.Sun_Net import SNUNet_ECAM
from torchvision.transforms.functional import to_tensor,normalize
from loss.focalloss import FocalLoss
from Models.unetPlusPlus import unetPlusPlus
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():

    base_lr = 0.01
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = unetPlusPlus(6,2).to(device, dtype=torch.float)

    optimizer = SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    criterion = FocalLoss(gamma=2, alpha=0.25)
    dataset = 'Gloucester-sar'
    model_name = dataset+'Unet++'
    save_result = 'result'
    model.load_state_dict(torch.load(('Weights/'+model_name+'.pth'),map_location=torch.device(device)))

    model.eval()
    loss_v = []
    test_TP = 0
    test_TN = 0
    test_FN = 0
    test_FP = 0
    i=0
    for f in os.listdir('./data/'+dataset+'/test/Image'):
        x1 = cv2.imread('./data/'+dataset+'/test/Image/' + f, 1).astype(float)
        x2 = cv2.imread('./data/'+dataset+'/test/Image2/' + f, 1).astype(float)
        lbl2 = cv2.imread('./data/'+dataset+'/test/label/' +f, 0)
        lbl2 = np.where(lbl2 > 0, 1, lbl2)
        x1 = to_tensor(np.ascontiguousarray(x1, dtype=np.float32))
        x1=x1.view(1,x1.shape[0],x1.shape[1],x1.shape[2])
        x2 = to_tensor(np.ascontiguousarray(x2, dtype=np.float32))
        x2 = x2.view(1,x2.shape[0],x2.shape[1], x2.shape[2])
        lbl = torch.tensor(lbl2)

        x1 = x1.to(device, dtype=torch.float)
        x2 = x2.to(device, dtype=torch.float)
        lbl = lbl.to(device, dtype=torch.long)

        x = torch.cat((x1, x2), dim=1)
        y3, y4, y5 = model(x)
        optimizer.zero_grad()

        lbl = lbl.float()
        lbl_4d = torch.unsqueeze(lbl, 0)
        lbl_4d = torch.unsqueeze(lbl_4d, 0)

        lbl3 = F.interpolate(lbl_4d, size=64, mode='nearest')
        lbl4 = F.interpolate(lbl_4d, size=128, mode='nearest')
        lbl5 = lbl_4d
        # lbl1 = torch.squeeze(lbl1,0).long()
        # lbl2 = torch.squeeze(lbl2,0).long()
        lbl3 = torch.squeeze(lbl3, 0).long()
        lbl4 = torch.squeeze(lbl4, 0).long()
        lbl5 = torch.squeeze(lbl5, 0).long()

        predict = torch.argmax(y5, 1)
        predict = predict.squeeze()

        loss3 = criterion(y3, lbl3)
        loss4 = criterion(y4, lbl4)
        loss5 = criterion(y5, lbl5)
        loss = 0.2 * loss3 + 0.3 * loss4 + 0.5 * loss5
        loss_v.append(loss.item())

#
        a = predict.cpu().detach().numpy().astype(np.uint8)
        a = np.where(a > 0, 255, a)
        a = Image.fromarray(a)
        a.save('./data/'+dataset+'/'+save_result+'/1/'+ f)

        b = np.where(lbl2 > 0, 255, lbl2)
        b = Image.fromarray(b)
        b.save('./data/'+dataset+'/'+save_result+'/2/'+ f)

        i=i+1
        test_TP += ((predict == 1).long() & (lbl == 1).long()).cpu().sum().numpy() + 0.001
        # TN    predict 和 label 同时为0
        test_TN += ((predict == 0).long() & (lbl == 0).long()).cpu().sum().numpy()
        # FN    predict 0 label 1
        test_FN += ((predict == 0).long() & (lbl == 1).long()).cpu().sum().numpy()
        # FP    predict 1 label 0
        test_FP += ((predict == 1).long() & (lbl == 0).long()).cpu().sum().numpy()
        # print("test:",TP, TN, FN, FP)
    precision = test_TP / (test_TP + test_FP)
    recall = test_TP / (test_TP + test_FN)
    f1 = 2 * recall * precision / (recall + precision)
    acc = (test_TP + test_TN) / (test_TP + test_TN + test_FP + test_FN)

    print("test_precition:", precision)
    print("test_recall:", recall)
    print("test_F1:", f1)
    print("test_ACC:", acc)
    print('test:', np.mean(loss_v))



if __name__ == '__main__':
    main()