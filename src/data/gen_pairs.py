import glob
import os.path
import numpy as np
import os
import re
'''
创建验证集bin的pairs.txt
'''
import random
# 图片数据文件夹
INPUT_DATA = '/home/huangju/dataset/QMUL-SurvFace/verification-set'
pairs_file_path = '/home/huangju/codes/insightface/datasets/pairs.txt'

rootdir_list = os.listdir(INPUT_DATA)
idsdir_list = [name for name in rootdir_list if os.path.isdir(os.path.join(INPUT_DATA, name))]
print(idsdir_list)

id_nums = len(idsdir_list)
print(id_nums)

def produce_same_pairs():
    matched_result = []  # 相同类的匹配对
    j=0
    while j < id_nums:
        #id_int= random.randint(0,id_nums-1)

        id_dir = os.path.join(INPUT_DATA, idsdir_list[j] )


        id_imgs_list = os.listdir(id_dir)

        id_list_len = len(id_imgs_list)
        # print(idsdir_list[j])
        if id_list_len>=2:
            id1_img_file = id_imgs_list[random.randint(0,id_list_len-1)]
            id2_img_file = id_imgs_list[random.randint(0,id_list_len-1)]
            while id2_img_file==id1_img_file:
                id2_img_file = id_imgs_list[random.randint(0,id_list_len-1)]

            id1_path = os.path.join(id_dir, id1_img_file)
            id2_path = os.path.join(id_dir, id2_img_file)

            same = 1
            #print([id1_path + '\t' + id2_path + '\t',same])
            matched_result.append((id1_path + '\t' + id2_path + '\t',same))
        print(j)
        j+=1
    return matched_result


def produce_unsame_pairs():
    unmatched_result = set()  # 不同类的匹配对
    # j=0
    # while j<20000:
    while len(unmatched_result)<6000:
        id1_int = random.randint(0,id_nums-1)
        id2_int = random.randint(0,id_nums-1)
        while id1_int == id2_int:
            id1_int = random.randint(0,id_nums-1)
            id2_int = random.randint(0,id_nums-1)

        id1_dir = os.path.join(INPUT_DATA, idsdir_list[id1_int])
        id2_dir = os.path.join(INPUT_DATA, idsdir_list[id2_int])

        id1_imgs_list = os.listdir(id1_dir)
        id2_imgs_list = os.listdir(id2_dir)
        id1_list_len = len(id1_imgs_list)
        id2_list_len = len(id2_imgs_list)
        if id1_list_len>0 and id2_list_len>0:

            id1_img_file = id1_imgs_list[random.randint(0, id1_list_len-1)]
            id2_img_file = id2_imgs_list[random.randint(0, id2_list_len-1)]

            id1_path = os.path.join(id1_dir, id1_img_file)
            id2_path = os.path.join(id2_dir, id2_img_file)

            same = 0
            unmatched_result.add((id1_path + '\t' + id2_path + '\t',same))
            print(len(unmatched_result))
            # j+=1
    return unmatched_result


same_result = produce_same_pairs()
print(same_result)
unsame_result = list(produce_unsame_pairs())

all_result = same_result + unsame_result

random.shuffle(all_result)
print(all_result)

file = open(pairs_file_path, 'w')
for line in all_result:
    file.write(line[0] + str(line[1]) + '\n')

file.close()