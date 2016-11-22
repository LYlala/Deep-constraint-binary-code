#! /usr/bin/env python
#coding=utf-8

'''
python examples/deepbinary/extract_feature.py \
data/256object/ref.txt \
examples/deepbinary/DCBC/deploy.prototxt \
examples/deepbinary/models/256object/256object_DCBC128_iter_30000.caffemodel \
data/ilsvrc12/imagenet_mean.npy  \
data/256object/feat128_DCBC_test/feat_binary


python examples/deepbinary/extract_feature.py \
data/256object/query.txt \
examples/deepbinary/DCBC/deploy.prototxt \
examples/deepbinary/models/256object/256object_DCBC128_iter_30000.caffemodel \
data/ilsvrc12/imagenet_mean.npy \
data/256object/feat128_DCBC_test/query


'''

import sys

sys.path.append("/opt/modules/caffe/python")
import caffe
import numpy as np
import os
import logging
import datetime

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print "usage: python compute_fea_for_image_retrieval.py [img_name_file] [net_def_prototxt] [trained_net_caffemodel] [image_mean_file] [out_put dir]"
		exit(1)

	batchsize = 1
	img_file = sys.argv[1]

	net_def_prototxt = sys.argv[2]
	
	trained_net_caffemodel = sys.argv[3]

	caffe.set_mode_gpu()

	net=caffe.Classifier(net_def_prototxt,trained_net_caffemodel,mean=np.load(sys.argv[4]).mean(1).mean(1),
		channel_swap=(2,1,0),
		raw_scale=255,
		image_dims=(256,256))

	if not os.path.exists(sys.argv[5]):
		os.makedirs(sys.argv[5])
	#if not os.path.exists(sys.argv[5]):
		#os.makedirs(sys.argv[5])
	#if not os.path.exists(sys.argv[6]):
		#os.makedirs(sys.argv[6])

	img_list=[]
	i=0
	for line in open(img_file):

		imgname, tag = line.strip().split(' ')
		#imgname= line.strip()

		filename=imgname.replace('/','_').replace('.jpg','')

		input_image=caffe.io.load_image('data/256object/256object/'+imgname)

		feat_binary=net.predict([input_image],False)

		#feat_binary = (feat_binary>=0.5)*1

		np.save(os.path.join(sys.argv[5], filename), feat_binary)
		i=i+1
		if i%100==0:
			print (str(i)+'has been storted')

