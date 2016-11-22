import numpy as np
import os
import sys
import glob
import time
import scipy.spatial.distance as distance
import numpy.matlib

feat_fc6=glob.glob('/opt/modules/caffe/data/corel/feat/feat_fc6/*.npy')
feat_fc7=glob.glob('/opt/modules/caffe/data/corel/feat/feat_fc7/*.npy')
feat_binary=glob.glob('/opt/modules/caffe/data/corel/feat/feat_binary/*.npy')

feat_retrieval_fc6=glob.glob('/opt/modules/caffe/data/corel/feat/feat_fc6/*0011.npy')
feat_retrieval_fc7=glob.glob('/opt/modules/caffe/data/corel/feat/feat_fc7/*0011.npy')
feat_retrieval_binary=glob.glob('/opt/modules/caffe/data/corel/feat/feat_binary/*0011.npy')

retrieval_Name=[x.split('\\')[-1].replace('.npy','') for x in feat_retrieval_fc6]

print "loading feature fc6"

score_fc6=np.zeros((len(retrieval_Name),len(feat_fc6)))
for i,q in enumerate(retrieval_Name):
    print i
    x=np.load(feat_retrieval_fc6[i])
    for j ,ref in enumerate(feat_fc6):
        y=np.load(feat_fc6[j])
        score_fc6[i][j]=distance.euclidean(x,y)


print "loading feature fc7"
start = time.time()
score_fc7=np.zeros((len(retrieval_Name),len(feat_fc7)))
for i,q in enumerate(retrieval_Name):
    print i
    x=np.load(feat_retrieval_fc7[i])
    for j ,ref in enumerate(feat_fc7):
        y=np.load(feat_fc7[j])
        score_fc7[i][j]=distance.euclidean(x,y)



print "loading feature binary"
start = time.time()
score_binary=np.zeros((len(retrieval_Name),len(feat_binary)))
for i,q in enumerate(retrieval_Name):
    print i
    x=np.load(feat_retrieval_binary[i])
    for j ,ref in enumerate(feat_binary):
        y=np.load(feat_binary[j])
        score_binary[i][j]=distance.euclidean(x[0],y[0])
end = time.time()
print end-start

print "caculate distance done"


labels=[]
for x in feat_fc7:
    labels.append(x.split('\\')[-1].split('_image_00')[0])
label=set(labels)

label_for_ref={}
for i, x in enumerate(label):
    label_for_ref[x]=i

score_sim=np.zeros((102,2040))
retrieval_label=[x.split('_image_00')[0] for x in retrieval_Name]
retrieval_mat=np.asarray([label_for_ref[x] for x in retrieval_label])

retrieval_mat=np.matlib.repmat(retrieval_mat,20,1).T
index_fc6=np.argsort(score_fc6,1)
index_fc7=np.argsort(score_fc7,1)
index_binary=np.argsort(score_binary,1)

ref_label=np.asarray([label_for_ref[x.split('\\')[-1].split('_image_00')[0]] for x in  feat_fc6])
res_label=np.matlib.repmat(ref_label,102,1)

res_fc6=[]
for i in xrange(102):
    res_fc6.append(res_label[i][index_fc6[i]])
res_fc6=np.asarray(res_fc6)[:,0:20]

res_fc7=[]
for i in xrange(102):
    res_fc7.append(res_label[i][index_fc7[i]])
res_fc7=np.asarray(res_fc7)[:,0:20]

res_binary=[]
for i in xrange(102):
    res_binary.append(res_label[i][index_binary[i]])
res_binary=np.asarray(res_binary)[:,0:20]

retrieval_res=(retrieval_mat==res_fc6)
res1=np.mean(np.sum(retrieval_res,1)/20.0)
print res1

retrieval_res=(retrieval_mat==res_fc7)
res2=np.mean(np.sum(retrieval_res,1)/20.0)
print res2

retrieval_res=(retrieval_mat==res_binary)
res3=np.mean(np.sum(retrieval_res,1)/20.0)
print res3





















