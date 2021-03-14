# -*- coding: utf-8 -*-
# @Time    : 2019/7/29 16:19
# @Author  : "梅俊辉"
# @Email   : 18211091722@163.com
# @File    : write_pairs.py 
# @Software: PyCharm

import os
import itertools

def txt_writer(path,txt_name):
    faces = []
    for root,_,files in os.walk(path):
        for fname in files:
            suffix = os.path.splitext(fname)[1].lower()
            if suffix in ['.jpg','.jpeg']:
                faces.append(os.path.join(root,fname))
    #print(faces)
    iter = list(itertools.combinations(faces, 2))
    print(len(faces))
    print(len(iter))
    txt = open(txt_name, "w")
    count=0
    for idx,(f1,f2) in enumerate(iter):
        if count%10000==0:
            print(count)
        if os.path.basename(os.path.split(f1)[0]) == \
                os.path.basename(os.path.split(f2)[0]):
            #iter[idx] = [f1,f2,1]
            txt.write(f1+','+f2+','+str(1)+'\n')
        else:
            txt.write(f1+','+f2+','+str(0)+'\n')
        count+=1
    print("finish1")
    txt.close()
    # with open(txt_name,'w') as txt:
    #     # txt.write(json.dumps(iter))
    #     for data in iter:
    #         txt.write(data[0]+','+data[1]+','+str(data[2])+'\n')
    # print("finish2")
    return iter

if __name__ == "__main__":
    txt_writer(r'/home/huangju/dataset/QMUL-SurvFace/align112',#数据目录
               r'/home/huangju/codes/insightface/datasets/pairs.txt'#生成文件目录
               )