import cv2
import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import numpy as np
from tqdm import tqdm

faces_path = r'/home/huangju/dataset/msmt17_112'          # 人脸数据文件夹
output_path = r'/home/huangju/dataset/msmt17_112_train'    # 对齐后的保存的人脸数据文件夹

for root,_,files in os.walk(faces_path):
    for fname in tqdm(files):
        if fname.split("-")[0]=="train":
            img = cv2.imread(os.path.join(root,fname))
            #img=cv2.resize(img,(112,112))
            new_id=int(fname.split("-")[1])+30000
            new_root = os.path.join(output_path,str(new_id))
        #print(new_root)
            if not os.path.exists(new_root):
                os.mkdir(new_root)
            new_fname=str(new_id)+'_'+'_'.join(fname.split("-")[-1].split('_')[1:]) 
            cv2.imwrite(os.path.join(new_root, new_fname), img)
        # cv2.imwrite(os.path.join(output_path, fname), img)
        # print(fname)


# for root,_,files in os.walk(faces_path):
#     for fname in tqdm(files):
#         img = cv2.imread(os.path.join(root,fname))
#         # img=cv2.resize(img,(112,112))
#         new_root = os.path.join(output_path,fname.split("_")[0])
#         #print(new_root)
#         if not os.path.exists(new_root):
#             os.mkdir(new_root)
        
#         cv2.imwrite(os.path.join(new_root, fname), img)
#         # cv2.imwrite(os.path.join(output_path, fname), img)
#         # print(fname)