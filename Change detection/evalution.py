import numpy as np
from PIL import Image
import os

cd_map_dir = "./data/Shuguang-sar-p2-aug/result_SUN/1"
label_dir = "./data/Shuguang-sar-p2-aug/result_SUN/2"
# cd_map_dir = "E:/code/PRCV-cpt/BIT_CD-master/result/PCRV512/1"
# label_dir = "E:/code/PRCV-cpt/BIT_CD-master/result/PCRV512/2"
if __name__ == "__main__":
    image_set1 = os.listdir(label_dir)
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(image_set1)):
        cd_map = Image.open(cd_map_dir+'/'+image_set1[i])
        label = Image.open(label_dir+'/'+image_set1[i])
        x = np.array(cd_map)
        y = np.array(label)
        #x = x[:, :, 0]
        #y = y[:,:,0]
        TP += ((x == 255) & (y == 255)).sum() + 0.0001
        # TN    predict 和 label 同时为0
        TN += ((x == 0) & (y == 0)).sum() + 0.0001
        # FN    predict 0 label 1
        FN += ((x == 0) & (y == 255)).sum() + 0.0001
        # FP    predict 1 label 0
        FP += ((x == 255) & (y == 0)).sum() + 0.0001
        print(TP,TN,FN,FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * recall * precision / (recall + precision)
    acc = (TP + TN) / (TP + TN + FP + FN)
    Miou = 0.5 * (TP / (TP + FP + FN)) + 0.5*(TN / (TN + FP + FN))
    print("test_precition:", precision)
    print("test_recall:", recall)
    print("test_F1:", f1)
    print("test_ACC:", acc)
    print("test_IOU:", Miou)
