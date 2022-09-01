import numpy as np
import os
import cv2
import torch
from torch import nn
from torch.optim import SGD,Adam
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from  PIL import Image
from utils.Ranger import Ranger
import torch.nn.functional as F
from loss.focalloss import FocalLoss
import time
from  siamunet_conc import SiamUnet_conc
from  siamunet_diff import SiamUnet_diff
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from torchvision.transforms.functional import to_tensor,normalize
from torch.autograd import Variable
#from Models.Sun_Net import SNUNet_ECAM
from Models.unetPlusPlus import unetPlusPlus

import logging
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class OSCD(Dataset):
    def __init__(self, dir_nm):
        super(OSCD, self).__init__()
        self.dir_nm = dir_nm
        self.file_ls = os.listdir(dir_nm+"/Image")
        self.file_size = len(self.file_ls)

    def __getitem__(self, idx):
        x1 = cv2.imread(self.dir_nm + '/Image/'+self.file_ls[idx],1).astype(float)
        x2 = cv2.imread(self.dir_nm + '/Image2/'+self.file_ls[idx],1).astype(float)
        # x1 = np.expand_dims(x1,0)
        # x2 = np.expand_dims(x2,0)
        x1 = np.transpose(x1,(2,0,1))
        x2 = np.transpose(x2,(2,0,1))
        lbl = cv2.imread(self.dir_nm + '/label/'+self.file_ls[idx], 0)
        lbl = np.where(lbl > 0, 1, lbl)
        return x1, x2, lbl

    def __len__(self):
        return self.file_size



def placeholder_file(path):
    """
    Creates an empty file at the given path if it doesn't already exists
    :param path: relative path of the file to be created
    """
    import os
    if not os.path.exists(path):
        with open(path, 'w'): pass
dataset = 'Gloucester'
model_name = dataset

def main():
    train_dir = './data/'+dataset+'/train'
    test_dir = './data/'+dataset+'/test'
    base_lr = 0.001
    warmup_epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = OSCD(train_dir)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_data = OSCD(test_dir)
    test_dataloader = DataLoader(test_data, batch_size=1,shuffle=True)
    #model = FC_EF().to(device, dtype=torch.float)

    #model = SiamUnet_conc(3,2).to(device, dtype=torch.float)
    #model = SiamUnet_diff(3,2).to(device, dtype=torch.float)
    model = unetPlusPlus(6,2).to(device, dtype=torch.float)
    # model = SNUNet_ECAM(3, 2).to(device, dtype=torch.float)
    # device_ids = [0, 1]
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    optimizer = Ranger(
        [
            {'params': [param for name, param in model.named_parameters()
                        if name[-4:] == 'bias'],
             'lr': 2 * base_lr},
            {'params': [param for name, param in model.named_parameters()
                        if name[-6:] == 'weight'],
             'lr': base_lr, 'weight_decay': 1e-4},
            {'params': [param for name, param in model.named_parameters()
                        if name[-6:] != 'weight' and name[-4:] != 'bias'],
             'lr': base_lr
             }], lr=base_lr)
    #scheduler =optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2,eta_min=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    #optimizer = SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    #optimizer = Adam(model.parameters(), lr=base_lr, weight_decay=0.0005)
    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2,alpha=0.25)
    val_best_loss=999
    best_F1 = 0.0
    no_optim=0
    EPOCH = 100
    Time = np.zeros(shape=200,dtype=np.float32)
    for epoch in range(EPOCH):
        time1 = time.time()
        P_v = []
        r_v = []
        F1_v = []
        acc_v = []
        loss_v = []
        model.train()
        if warmup_epochs and (epoch + 1) < warmup_epochs:
            warmup_percent_done = (epoch + 1) / warmup_epochs
            warmup_learning_rate = base_lr * warmup_percent_done  # gradual warmup_lr
            learning_rate = warmup_learning_rate
            optimizer.param_groups[0]['lr'] = learning_rate * 2
            optimizer.param_groups[1]['lr'] = learning_rate
        elif warmup_epochs and (epoch + 1) == warmup_epochs:
            optimizer.param_groups[0]['lr'] = base_lr
            optimizer.param_groups[1]['lr'] = base_lr
        print('Epoch:[{:0>3}] '.format(epoch + 1) + 'Learning rate ' + str(optimizer.param_groups[1]['lr']))
        for i, data in enumerate(train_dataloader):
            x1, x2, lbl = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device,dtype=torch.long)
            x = torch.cat((x1,x2),dim=1)
            y3,y4,y5 = model(x)

            optimizer.zero_grad()

            lbl = lbl.float()
            lbl_4d = torch.unsqueeze(lbl,1)

            lbl3 = F.interpolate(lbl_4d,size=64, mode='nearest')
            lbl4 = F.interpolate(lbl_4d, size=128, mode='nearest')
            lbl5 = lbl_4d

            lbl3 = torch.squeeze(lbl3).long()
            lbl4 = torch.squeeze(lbl4).long()
            lbl5 = torch.squeeze(lbl5).long()
            loss3 = criterion(y3, lbl3)
            loss4 = criterion(y4, lbl4)
            loss5 = criterion(y5, lbl5)

            loss = 0.2*loss3+0.3*loss4+0.5*loss5

            loss.backward()
            optimizer.step()

            predict = torch.argmax(y5, 1)
            predict = predict.squeeze()
            loss_v.append(loss.item())
            if(i%50==0 and i>0):
                print(np.mean(loss_v))

            lbl = lbl.long()
            TP = ((predict == 1).long()& (lbl == 1).long()).cpu().sum().numpy()+0.0001
            # TN    predict 和 label 同时为0
            TN = ((predict == 0).long() & (lbl == 0).long()).cpu().sum().numpy()+0.0001
            # FN    predict 0 label 1
            FN = ((predict == 0).long() & (lbl == 1).long()).cpu().sum().numpy()+0.0001
            # FP    predict 1 label 0
            FP = ((predict == 1).long() & (lbl == 0).long()).cpu().sum().numpy()+0.0001
            #print(TP, TN, FN, FP)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * recall * precision / (recall + precision)
            acc = (TP + TN) / (TP + TN + FP + FN)

            P_v.append(precision.item())
            r_v.append(recall.item())
            F1_v.append(f1.item())
            acc_v.append(acc.item())

        print("precition:", np.mean(P_v))
        print("recall:", np.mean(r_v))
        print("F1:", np.mean(F1_v))
        print("acc:", np.mean(acc_v))
        print("loss:",np.mean(loss_v))

        loss_v = []
        model.eval()
        test_TP = 0
        test_TN = 0
        test_FP = 0
        test_FN = 0
        for i, data in enumerate(test_dataloader):
            x1, x2, lbl = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)

            x = torch.cat((x1, x2), dim=1)
            y3, y4, y5 = model(x)

            optimizer.zero_grad()

            lbl = lbl.float()
            lbl_4d = torch.unsqueeze(lbl, 0)

            lbl3 = F.interpolate(lbl_4d, size=64, mode='nearest')
            lbl4 = F.interpolate(lbl_4d, size=128, mode='nearest')
            lbl5 = lbl_4d

            lbl3 = torch.squeeze(lbl3,0).long()
            lbl4 = torch.squeeze(lbl4,0).long()
            lbl5 = torch.squeeze(lbl5,0).long()


            loss3 = criterion(y3, lbl3)
            loss4 = criterion(y4, lbl4)
            loss5 = criterion(y5, lbl5)
            loss = 0.2*loss3+0.3*loss4+0.5*loss5


            predict = torch.argmax(y5, 1)
            lbl = lbl.squeeze()
            loss_v.append(loss.item())
            lbl = lbl.long()
            test_TP += ((predict == 1).long() & (lbl == 1).long()).cpu().sum().numpy()+0.001
            # TN    predict 和 label 同时为0
            test_TN += ((predict == 0).long() & (lbl == 0).long()).cpu().sum().numpy()
            # FN    predict 0 label 1
            test_FN += ((predict == 0).long() & (lbl == 1).long()).cpu().sum().numpy()
            # FP    predict 1 label 0
            test_FP += ((predict == 1).long() & (lbl == 0).long()).cpu().sum().numpy()
        precision = test_TP / (test_TP + test_FP)
        recall = test_TP / (test_TP + test_FN)
        f1 = 2 * recall * precision / (recall + precision)
        acc = (test_TP + test_TN) / (test_TP + test_TN + test_FP + test_FN)

        print("test_precition:", precision)
        print("test_recall:", recall)
        print("test_F1:", f1)
        print("test_ACC:", acc)
        print('test:', np.sum(loss_v))

        if f1 <= best_F1:
            no_optim += 1
        else:
            no_optim = 0
            best_F1 = f1
            #val_best_loss = valloss
            if epoch != 0:
                print('Saving model!')
                torch.save(model.state_dict(), 'Weights/'+model_name+'.pth')
        if no_optim > 50:
            print('early stop at %d epoch' % epoch)
            break
        if no_optim >= 5:
            if float(scheduler.get_last_lr()[-1]) < 1e-7:
                break
            print('Loading model')
            if os.path.exists('Weights/'+model_name+'.pth'):
                model.load_state_dict(torch.load('Weights/'+model_name+'.pth'))
            else:
                print('No saved Model! Loading Init Model!')
                model.load_state_dict(torch.load('Weights/'+model_name+'.pth'))
            scheduler.step()
            no_optim = 0
        time2 = time.time()
        A = time2-time1
        Time[epoch] = A
        print("Times:",Time.sum()/(epoch+1))

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    placeholder_file('Weights/last.pth')
    torch.save(model.state_dict(), 'Weights/last.pth')

    placeholder_file('Weights/' + current_datetime + "-" + '.pth')
    torch.save(model.state_dict(), 'Weights/' + current_datetime + "-" + '.pth')
    logging.info(f'Model saved')

if __name__ == '__main__':
    main()