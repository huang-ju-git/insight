import cv2
import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import numpy as np

faces_path = r'/home/huangju/dataset/face_imgs_detected_larger'          # 人脸数据文件夹
output_path = r'/home/huangju/dataset/msmt17_112'    # 对齐后的保存的人脸数据文件夹

for root,_,files in os.walk(faces_path):
    for fname in files:
        img = cv2.imread(os.path.join(root,fname))
        img=cv2.resize(img,(112,112))
        # new_root = os.path.join(output_path,fname.split("_")[0])
        # print(new_root)
        # if not os.path.exists(new_root):
        #     os.mkdir(new_root)
        
        # cv2.imwrite(os.path.join(new_root, fname), img)
        cv2.imwrite(os.path.join(output_path, fname), img)
        print(fname)