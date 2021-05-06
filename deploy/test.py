import face_model
import argparse
import cv2
import sys
import numpy as np
import pickle
import os
import random
from numpy import linalg
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='1'

args = None

def cos_sim(coords1:np.array,coords2:np.array):

    num = sum(coords1 * coords2)  # 若为行向量则 A * B.T
    denom = linalg.norm(coords1) * linalg.norm(coords2)
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim


def parse_args(path):

    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    #人脸识别模型
    parser.add_argument('--model', default=path, help='path to load model.')
    #年龄识别模型
    parser.add_argument('--ga-model', default='/home/huangju/codes/insightface/models/gamodel-r50/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    args = parser.parse_args()
    return args

#示例用法，获取两幅图像特征并获取相似度
# model = face_model.FaceModel(args)
# img = cv2.imread('/home/huangju/dataset/face_imgs_detected/test-2234-2234_004_01_0302afternoon_0748_0.jpg')
# print("size: {}".format(img.shape))
# img = model.get_input(img)
# print("img: {}".format(img))
# f1 = model.get_feature(img)
# print(f1)
# print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
# img = cv2.imread('/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
# f2 = model.get_feature(img)
# dist = np.sum(np.square(f1-f2))
# print(dist)
# sim = np.dot(f1, f2.T)
# print(sim)
# #diff = np.subtract(source_feature, target_feature)
# #dist = np.sum(np.square(diff),1)

def test(args):
    #获取所有人脸特征，并存到pkl文件中
    model = face_model.FaceModel(args)
    pathss=[]
    for root, dirs, files in os.walk("/home/huangju/dataset/face_imgs_detected_larger"):
        path = [os.path.join(root, name) for name in files]
            #print(path)
        pathss.extend(path)
    #print(pathss)
    #print("finish")
    count=0 #记录提取特征的人脸的图片数量
    d={} #key: path  value: feature
    for path in tqdm(pathss):
        if path.split('/')[-1].split("-")[0]=="train" or path.split('/')[-1].split("-")[0]=="query":
            continue
        img = cv2.imread(path)
        img = model.get_input(img)
        if img is not None:
            feature = model.get_feature(img)
            key="data/msmt17/test/"+path.split('/')[-1].split('-')[1]+"/"+path.split('/')[-1].split('-')[-1]
            d[key]=feature
            count+=1
            #print("count: {}".format(count))
    

    # with open("msmt_face_q_t-own2.pkl","wb") as f: 
    #     pickle.dump(d,f)
    with open("face_feature-surv_ms1m_msmt.pkl","wb") as f: 
        pickle.dump(d,f)

def cal():
    #读取人脸特征
    face_features=pickle.load(open('/home/huangju/codes/insightface/deploy/msmt_face-best.pkl', 'rb'))
    # print(face_features.keys())
    print(len(face_features.keys()))
    query_face=pickle.load(open('/home/huangju/codes/tracking_system-master/src/query_face.pkl', 'rb'))
    query_face_own={}


    #计算query和gallery中每张图像的similarity
    for term in query_face.keys():
        query_face[term]=face_features[term]
        del face_features[term]
    print(len(face_features.keys()))
    similarities_f={} #key:query  value: dict(key:face  value:sim)
    i=0
    for term in query_face.keys():
        sims={}
        for face in face_features.keys():
            sim=cos_sim(np.array(query_face[term]),np.array(face_features[face]))
            # print(sim)
            sims[face]=sim
        similarities_f[term]=sims
        #print(i)
        i+=1
    with open("similarities_f-own.pkl","wb") as f: 
        pickle.dump(similarities_f,f)

    similarities_f=pickle.load(open('similarities_f-own.pkl', 'rb'))

    #计算rank-1
    count=0
    for term in similarities_f.keys():
        d=similarities_f[term]
        face=max(d, key=lambda k: d[k])
        if term.split('/')[3]==face.split('/')[3]:
            count+=1
        # if term.split('/')[0]=="train":
        #     if face.split('-')[0]=="train" and term.split('-')[1]==face.split('-')[1]:
        #         count+=1
        # else:
        #     if face.split('-')[0]!="train" and term.split('-')[1]==face.split('-')[1]:
        #         count+=1
    print("only face rank-1:")
    print(count/2000)
    face_rank1=count/2000
    return face_rank1

#对于小范围人脸，计算只缩放不矫正的效果
def cal_small():
    #读取人脸特征
    face_features=pickle.load(open('/home/huangju/codes/insightface/deploy/msmt_face_q_t-own-best.pkl', 'rb')) #4w张图像
    face_features_small=pickle.load(open("/home/huangju/codes/insightface/deploy/msmt_face_q_t-own.pkl", 'rb')) #2w张图像
    # print(face_features.keys())
    print(len(face_features_small.keys()))
    query_face=pickle.load(open('/home/huangju/codes/tracking_system-master/src/query_face.pkl', 'rb'))
    query_face_own={}


    #计算query和gallery中每张图像的similarity
    for term in query_face.keys():
        query_face[term]=face_features[term]
        del face_features_small[term]
    print(len(face_features_small.keys()))
    similarities_f={} #key:query  value: dict(key:face  value:sim)
    i=0
    for term in query_face.keys():
        sims={}
        for face in face_features_small.keys(): #使用小范围key
            sim=cos_sim(np.array(query_face[term]),np.array(face_features[face])) #使用未经矫正处理的特征
            # print(sim)
            sims[face]=sim
        similarities_f[term]=sims
        #print(i)
        i+=1
    # with open("similarities_f-own-best.pkl","wb") as f: 
    #     pickle.dump(similarities_f,f)

    # similarities_f=pickle.load(open('similarities_f-own-best.pkl', 'rb'))

    #计算rank-1
    count=0
    for term in similarities_f.keys():
        d=similarities_f[term]
        face=max(d, key=lambda k: d[k])
        if term.split('/')[3]==face.split('/')[3]:
            count+=1
        # if term.split('/')[0]=="train":
        #     if face.split('-')[0]=="train" and term.split('-')[1]==face.split('-')[1]:
        #         count+=1
        # else:
        #     if face.split('-')[0]!="train" and term.split('-')[1]==face.split('-')[1]:
        #         count+=1
    print("only face rank-1:")
    print(count/2000)
    face_rank1=count/2000
    return face_rank1


if __name__ == '__main__':

    # ----------------------逐个测试各个模型在msmt17上的rank-1------------------------
    # default_path='/home/huangju/codes/insightface/recognition/ArcFace/models6/r100-arcface-emore/model,12'
    # base_path='/home/huangju/codes/insightface/recognition/ArcFace/models8-3/r100-softmax-emore/model,'
    # #base_path='/home/huangju/codes/insightface/models/model-r100-ii/model,'
    # i=54
    # res={}
    # while i<55:   #45-60
    #     print("params "+str(i))
    #     path=base_path+str(i)
    #     args = parse_args(path)
    #     test(args)
    #     rank1=cal()
    #     res[i]=rank1
    #     print(res)
    #     i+=1
    # print(res)

    # ---------------------对于小范围人脸，计算只缩放不矫正的效果----------------------
    # rank1=cal_small()
    # print(rank1)


    
    base_path='/home/huangju/codes/insightface/models/model-r100-ii/model,'
    # base_path='/home/huangju/codes/insightface/recognition/ArcFace/models-surv_ms1m_based_msmt17-align-softmax/r100-softmax-emore/model,'
    i=0
    path=base_path+str(i)
    args = parse_args(path)
    test(args)
    
    


