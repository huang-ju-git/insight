import glob
import os.path as osp

dir_path="/home/huangju/dataset/dml_ped12"
img_paths = glob.glob(osp.join(dir_path, '*/*/*.jpg'))
#print(len(img_paths))
count=0
for img_path in img_paths:
    #print(img_path.split('/')[-1].split("_")[-1])
    if img_path.split('/')[-1].split('.')[0].split("_")[-1]=="face":
        count+=1
        new_key=img_path.split('/')[-3]+"/"+img_path.split('/')[-2]+"/"+img_path.split('/')[-1]
        #print(new_key)
print(count)