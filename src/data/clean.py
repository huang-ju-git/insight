#删除align112目录下的空文件夹，并获取非空类别总数

import os
# def del_emp_dir(path):
#     for (root, dirs, files) in os.walk(path):
#         for item in dirs:
#             dir = os.path.join(root, item)
#             try:
#                 os.rmdir(dir)  #os.rmdir() 方法用于删除指定路径的目录。仅当这文件夹是空的才可以, 否则, 抛出OSError。
#                 print(dir)
#             except Exception as e:
#                 print('Exception',e)

def count_dir(path):
    
    for (root, dirs, files) in os.walk(path):
        print(root)
        for item in files:
            print(item)
    

if __name__ == '__main__':
    # dir = r'/home/huangju/dataset/QMUL-SurvFace/verification_images/'
    # count_dir(dir)
    id1_path='/home/huangju'
    id2_path='QMUL-SurvFace/verification_images'
    same=1
    match=set()
    match.add((id1_path + '\t' + id2_path + '\t',same))
    match.add((id1_path + '\t' + id2_path + '\t',same))
    match.add(2)
    
    print(match)
    print(list(match))
    print("\nsame is ", same)
    print("\nsame is ", same)