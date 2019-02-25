"""
This file is a core part for extracting patches from cases.
The terms 'train' and 'test' are used to distinguish whether it have labels or not.
"""
import numpy as np
import medpy.io as medio
import os, sys, settings
from glob import glob
from sklearn.feature_extraction.image import extract_patches
from multiprocessing import Pool
from utils import *
import logging

MULTI = settings.MULTI

def process(args):
    fun, arg = args
    img_tmp, label_tmp, row_size_tmp, channel_size_tmp, num_patch_tmp, n_time_point_tmp, proportion_tmp = arg
    
    return fun(img_tmp, label_tmp, row_size_tmp, channel_size_tmp, num_patch_tmp, n_time_point_tmp, proportion_tmp)


def load_mri_from_directory(data_index_list, fixed_width, fixed_depth, is_test=False, data_dir='../input/train_data', is_fixed_size=True):
	img_train_list, img_label_list=[], []
	subject_list = sorted(glob(data_dir+'/*'))

	for data_index in data_index_list:
		img_list = [fn for fn in sorted(glob( os.path.join(subject_list[data_index], '*', '*.nii*')))  \
		if '4DPWI' not in fn and 'OT' not in fn]
		img_train=[]

		#Load images
		for img in img_list:
			test = (medio.load(img)[0])
			if is_fixed_size is True:
				test = transform_to_fixed_size(test, fixed_width, fixed_depth)
			test = normalize_img(test)
			img_train.append(test)
		img_train_list.append((img_train))

		if is_test is not True:
			label_path = glob(os.path.join(subject_list[data_index], '*', '*OT*.nii*'))[0]
			label = medio.load(label_path)[0]
			if is_fixed_size is True:
				label = transform_to_fixed_size(label, fixed_width, fixed_depth)
			img_label_list.append(label)

	if is_test is not True:
		return img_train_list, img_label_list
	else:
		return img_train_list


def extract_multiscale_patches_from_mri(data_index_list, data_dir, is_test=False, is_oversampling=True, \
								row_size=16, channel_size=8, num_patch=100, proportion=0.1, \
								addClinical=False, patch_r_stride=4, patch_c_stride=4, \
								is_fixed_size=True, n_time_point=6, fixed_width=256, fixed_depth=32):
	"""
	is_oversampling : indicator for oversampling patches near the lesion, this deicides whether 
	'extract_train_core_multscale' is used or not
	is_test : indicator for extract y_label or not. Extract y_label when 'is_test' is false.
	"""

	if is_test is not True:
		img_list, label_list=load_mri_from_directory(data_index_list, fixed_width, fixed_depth, \
			is_test=is_test, data_dir=data_dir, is_fixed_size=is_fixed_size)
	else:
		img_list=load_mri_from_directory(data_index_list, fixed_width, fixed_depth, \
			is_test=is_test, data_dir=data_dir, is_fixed_size=is_fixed_size)

	n=len(img_list)
	if is_test is not True:
		"""
		Extract patches for training and validation time
		"""

		args = []
		if is_oversampling is True:
			for k in xrange(n):
				args.append((extract_train_core_multiscale, \
					(img_list[k], label_list[k], row_size, channel_size, num_patch, n_time_point, proportion)))
		else:
			for k in xrange(n):
				args.append((extract_val_core_multiscale, \
					(img_list[k], label_list[k], row_size, channel_size, num_patch, n_time_point, proportion)))

		pool = Pool(processes=MULTI)
		batchsize = MULTI
		batches = n // batchsize + (n % batchsize != 0)

		logging.info('pooling : begin') 
		result=[]
		for k in xrange(batches):
			logging.info("batch {:>2} / {}".format(k + 1, batches))
			result.extend(pool.map(process, args[k * batchsize: (k + 1) * batchsize]))
		pool.close()
		logging.info('pooling : done')

		X_main=[]
		X_aug=[]
		Y=[]
		lesion_dicator_list=[]

		logging.info('number of extracted cases are :{}'.format(len(result)))
		for j in xrange(len(result)):
			X_main = X_main + result[j][0]
			X_aug = X_aug + result[j][1]
			Y = Y + result[j][2]
			lesion_dicator_list = lesion_dicator_list + result[j][3]

		X_main, X_aug, Y, lesion_dicator_list = np.array(X_main).astype('float32'), np.array(X_aug).astype('float32'), \
										np.array(Y), np.array(lesion_dicator_list)	
		
		if addClinical is True:
			pass
		else:
			return X_main, X_aug, Y, lesion_dicator_list

	else:
		"""
		Extract patches for test time
		"""
		X_main = []
		X_aug = []        
		img_shape_cache = []
		clinical_list = []
		for k in xrange(n):
			x_main=[]
			x_aug=[]            
			for time_point in xrange(n_time_point):
				#Load image
				img = img_list[k][time_point]
				img_aug = np.ones(img.shape + np.array([row_size, row_size, channel_size]))*img[0,0,0]
				img_aug[(row_size/2):(img.shape[0] + row_size/2), \
					(row_size/2):(img.shape[1] + row_size/2), \
					(channel_size/2):(img.shape[2] + channel_size/2)] = img
				img_aug = transform_shrink(img_aug)

                #Transpose
				img = np.transpose(np.array(img), (2, 0, 1))
				img_aug = np.transpose(np.array(img_aug), (2, 0, 1))

                #Make patches
				patches = extract_patches(img, (channel_size, row_size, row_size), \
					extraction_step=(patch_c_stride, patch_r_stride, patch_r_stride))
				x_main.append(patches)
				patches_aug = extract_patches(img_aug, (channel_size, row_size, row_size), \
					extraction_step=(patch_c_stride/2, patch_r_stride/2, patch_r_stride/2))
				x_aug.append(patches_aug)

			N_patches = np.prod(patches.shape)//(channel_size*(row_size**2))
			X_main.append(np.array(x_main))
			X_aug.append(np.array(x_aug))            
			img_shape_cache.append(img.shape)

		if addClinical is True:
			pass
		else:
			return X_main, X_aug, img_shape_cache
	


def extract_train_core_multiscale(img_raw, label, row_size, channel_size, num_patch, n_time_point, proportion):
	"""
	This code is to extract more patches near the lesion, and to augment saggital reflections.
	"""
	Y=[]
	X_main=[]
	X_aug=[]
	lesion_dicator_list=[]
	w, _, c=label.shape

	m1=np.where(np.sum(label, axis=1)>0)[0] #  np.where(np.sum(label, axis=(1,2))>0)[0]
	m2=np.where(np.sum(label, axis=0)>0)[0] #  np.where(np.sum(label, axis=(0,2))>0)[0]
	m3=np.where(np.sum(label, axis=1)>0)[1] #  np.where(np.sum(label, axis=(0,1))>0)[0]
	patch_dim=(row_size**2)*channel_size

	N_GT_patches=int(num_patch*(proportion))
	N_random_patches=num_patch-N_GT_patches

	image_shrink_list=[transform_shrink(image_tmp) for image_tmp in img_raw]

	for ind in zip( np.random.choice(range(make_range(min(m1), w, row_size), \
		make_range(max(m1), w, row_size, False)), N_GT_patches), \
		            np.random.choice(range(make_range(min(m2), w, row_size), \
		                make_range(max(m2), w, row_size, False)), N_GT_patches), \
		            np.random.choice(range(make_range(min(m3), c, channel_size), \
		                make_range(max(m3), c, channel_size, False)), N_GT_patches)):

		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list_aug = np.array(ind)//2 + np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list = tuple(map(tuple, ind_list))
		ind_list_aug = tuple(map(tuple, ind_list_aug))
		y = np.array([label[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		lesion_dicator = (np.sum(y) > 0)

		x_main=[]
		x_aug=[]
		x_saggital=[]
		x_saggital_aug=[]
		for m in xrange(n_time_point):
			img = np.array([(img_raw[m])[idx] \
			    for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x_main.append(img)
			x_saggital.append(transform_saggital(img))

			temp = np.array([(image_shrink_list[m])[idx] \
			    for idx in ind_list_aug]).reshape(row_size,row_size,channel_size)

			x_aug.append(temp)
			x_saggital_aug.append(transform_saggital(temp))

		Y.append(y)
		X_main.append(np.array(x_main))
		X_aug.append(np.array(x_aug))
		lesion_dicator_list.append(lesion_dicator)

		Y.append(transform_saggital(y))
		X_main.append(np.array(x_saggital))
		X_aug.append(np.array(x_saggital_aug))
		lesion_dicator_list.append(lesion_dicator)

	for ind in zip(np.random.choice(w-2*row_size, N_random_patches), \
		               np.random.choice(w-2*row_size, N_random_patches), \
		               np.random.choice(c-2*channel_size, N_random_patches) ) :

		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list_aug = np.array(ind)//2 + np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list = tuple(map(tuple, ind_list))
		ind_list_aug = tuple(map(tuple, ind_list_aug))
		y=np.array([label[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		lesion_dicator=(np.sum(y) > 0)
		
		x_main=[]
		x_aug=[]            
		x_saggital=[]
		x_saggital_aug=[]
		for m in xrange(n_time_point):
			img=np.array([(img_raw[m])[idx] \
			    for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x_main.append(img)
			x_saggital.append(transform_saggital(img))

			temp = np.array([(image_shrink_list[m])[idx] \
			    for idx in ind_list_aug]).reshape(row_size,row_size,channel_size)

			x_aug.append(temp)
			x_saggital_aug.append(transform_saggital(temp))

		Y.append(y)
		X_main.append(np.array(x_main))
		X_aug.append(np.array(x_aug))
		lesion_dicator_list.append(lesion_dicator)
		
		Y.append(transform_saggital(y))
		X_main.append(np.array(x_saggital))
		X_aug.append(np.array(x_saggital_aug))
		lesion_dicator_list.append(lesion_dicator)
		
	return X_main, X_aug, Y, lesion_dicator_list


def extract_val_core_multiscale(img_raw, label, row_size, channel_size, num_patch, n_time_point, *unused_params):
	Y = []
	X_main = []
	X_aug = []  
	lesion_dicator_list = []      
	w, _, c = img_raw[0].shape
	patch_dim=(row_size**2)*channel_size

	image_shrink_list=[transform_shrink(image_tmp) for image_tmp in img_raw]

	for ind in zip( np.random.choice(w-2*row_size, num_patch), \
	                np.random.choice(w-2*row_size, num_patch), \
	                np.random.choice(c-2*channel_size, num_patch)):
		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list_aug = np.array(ind)//2 + np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list = tuple(map(tuple, ind_list))
		ind_list_aug = tuple(map(tuple, ind_list_aug))
		y = np.array([label[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		lesion_dicator = (np.sum(y) > 0)

		x_main = []
		x_aug = []                
		for m in xrange(n_time_point):
			img = np.array([(img_raw[m])[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x_main.append(img)

			temp = np.array([(image_shrink_list[m])[idx] for idx in ind_list_aug]).reshape(row_size,row_size,channel_size)
			x_aug.append(temp)

		Y.append(y)
		X_main.append(np.array(x_main))
		X_aug.append(np.array(x_aug))
		lesion_dicator_list.append(lesion_dicator)

	return X_main, X_aug, Y, lesion_dicator_list


def extract_patches_from_mri(data_index_list, data_dir, is_test=False, is_oversampling=True, \
								row_size=16, channel_size=8, num_patch=100, proportion=0.1, \
								addClinical=False, patch_r_stride=4, patch_c_stride=4, \
								is_fixed_size=True, n_time_point=6, fixed_width=256, fixed_depth=32):
	"""
	is_oversampling : indicator for oversampling patches near the lesion, this deicides whether 
	'extract_train_core' is used or not
	is_test : indicator for extract y_label or not. Extract y_label when 'is_test' is false.
	"""

	if is_test is not True:
		img_list, label_list=load_mri_from_directory(data_index_list, fixed_width, fixed_depth, \
			is_test=is_test, data_dir=data_dir, is_fixed_size=is_fixed_size)
	else:
		img_list=load_mri_from_directory(data_index_list, fixed_width, fixed_depth, \
			is_test=is_test, data_dir=data_dir, is_fixed_size=is_fixed_size)

	n=len(img_list)
	if is_test is not True:
		"""
		Extract patches for training and validation time
		"""
		args = []
		if is_oversampling is True:
			for k in xrange(n):
				args.append((extract_train_core, \
					(img_list[k], label_list[k], row_size, channel_size, num_patch, n_time_point, proportion)))
		else:
			for k in xrange(n):
				args.append((extract_val_core, \
					(img_list[k], label_list[k], row_size, channel_size, num_patch, n_time_point, proportion)))

		pool = Pool(processes=MULTI)
		batchsize = MULTI
		batches = n // batchsize + (n % batchsize != 0)

		logging.info('pooling : begin') 
		result=[]
		for k in xrange(batches):
			logging.info("batch {:>2} / {}".format(k + 1, batches))
			result.extend(pool.map(process, args[k * batchsize: (k + 1) * batchsize]))
		pool.close()
		logging.info('pooling : done')

		X_main=[]
		Y=[]
		lesion_dicator_list=[]
		for j in xrange(len(result)):
			X_main = X_main + result[j][0]
			Y = Y + result[j][1]
			lesion_dicator_list = lesion_dicator_list + result[j][2]

		X_main, Y, lesion_dicator_list = np.array(X_main).astype('float32'),  \
										np.array(Y), np.array(lesion_dicator_list)
		
		if addClinical is True:
			pass
		else:
			return X_main, Y, lesion_dicator_list

	else:
		"""
		Extract patches for test time
		"""
		X_main = []
		img_shape_cache = []
		clinical_list = []
		for k in xrange(n):
			x_main=[]
			for time_point in xrange(n_time_point):
				#Load image
				img=img_list[k][time_point]
                #Transpose
				img=np.transpose(np.array(img), (2, 0, 1))
                #Make patches
				patches=extract_patches(img, (channel_size, row_size, row_size), \
					extraction_step=(patch_c_stride, patch_r_stride, patch_r_stride))
				x_main.append(patches)

			N_patches = np.prod(patches.shape)//(channel_size*(row_size**2))
			X_main.append(np.array(x_main))
			img_shape_cache.append(img.shape)

		if addClinical is True:
			pass
		else:
			return X_main, img_shape_cache
	

def extract_train_core(img_raw, label, row_size, channel_size, num_patch, n_time_point, proportion):
	"""
	This code is to extract more patches near the lesion, and to augment saggital reflections.
	"""
	Y=[]
	X_main=[]
	lesion_dicator_list=[]
	w, _, c=label.shape
  
	m1=np.where(np.sum(label, axis=1)>0)[0]
	m2=np.where(np.sum(label, axis=0)>0)[0]
	m3=np.where(np.sum(label, axis=1)>0)[1]
	patch_dim=(row_size**2)*channel_size

	N_GT_patches=int(num_patch*(proportion))
	N_random_patches=num_patch-N_GT_patches

	for ind in zip( np.random.choice(range(make_range(min(m1), w, row_size), \
		make_range(max(m1), w, row_size, False)), N_GT_patches), \
		            np.random.choice(range(make_range(min(m2), w, row_size), \
		                make_range(max(m2), w, row_size, False)), N_GT_patches), \
		            np.random.choice(range(make_range(min(m3), c, channel_size), \
		                make_range(max(m3), c, channel_size, False)), N_GT_patches)):

		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list = tuple(map(tuple, ind_list))
		y = np.array([label[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		lesion_dicator = (np.sum(y) > 0)

		x_main=[]
		x_saggital=[]
		for m in xrange(n_time_point):
			img = np.array([(img_raw[m])[idx] \
			    for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x_main.append(img)
			x_saggital.append(transform_saggital(img))

		Y.append(y)
		X_main.append(np.array(x_main))
		lesion_dicator_list.append(lesion_dicator)

		Y.append(transform_saggital(y))
		X_main.append(np.array(x_saggital))
		lesion_dicator_list.append(lesion_dicator)

	for ind in zip(np.random.choice(w-2*row_size, N_random_patches), \
		               np.random.choice(w-2*row_size, N_random_patches), \
		               np.random.choice(c-2*channel_size, N_random_patches) ) :

		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list = tuple(map(tuple, ind_list))
		y=np.array([label[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		lesion_dicator=(np.sum(y) > 0)
		
		x_main=[]
		x_saggital=[]
		for m in xrange(n_time_point):
			img=np.array([(img_raw[m])[idx] \
			    for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x_main.append(img)
			x_saggital.append(transform_saggital(img))
			
		Y.append(y)
		X_main.append(np.array(x_main))
		lesion_dicator_list.append(lesion_dicator)
		
		Y.append(transform_saggital(y))
		X_main.append(np.array(x_saggital))
		lesion_dicator_list.append(lesion_dicator)
		
	return X_main, Y, lesion_dicator_list


def extract_val_core(img_raw, label, row_size, channel_size, num_patch, n_time_point, *unused_params):
	Y = []
	X_main = []
	lesion_dicator_list = []      
	w, _, c = img_raw[0].shape
	patch_dim=(row_size**2)*channel_size

	for ind in zip( np.random.choice(w-2*row_size, num_patch), \
	                np.random.choice(w-2*row_size, num_patch), \
	                np.random.choice(c-2*channel_size, num_patch)):
		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list = tuple(map(tuple, ind_list))
		y = np.array([label[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		lesion_dicator = (np.sum(y) > 0)
		x_main = []
		for m in xrange(n_time_point):
			img = np.array([(img_raw[m])[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x_main.append(img)
		Y.append(y)
		X_main.append(np.array(x_main))
		lesion_dicator_list.append(lesion_dicator)

	return X_main, Y, lesion_dicator_list	


def make_brain_from_patches(y_val_patch_pred_tmp, cache_tmp, patch_r_stride=4, patch_c_stride=4):
	c, h, w = cache_tmp
	y_val_patch_pred_tmp = y_val_patch_pred_tmp.squeeze()
	y_val_patch_pred = np.zeros(cache_tmp)
	N, c_patch, h_patch, w_patch = y_val_patch_pred_tmp.shape

	c1=int((c-c_patch)/patch_c_stride) + 1
	h1=int((h-h_patch)/patch_r_stride) + 1    
	w1=int((w-w_patch)/patch_r_stride) + 1
    
	for i in xrange(N):
		a,b,c = (i/(h1*w1), (i/w1)%h1, i%w1)
		y_val_patch_pred[(patch_c_stride*a):(patch_c_stride*a+c_patch), \
		(patch_r_stride*b):(patch_r_stride*b+h_patch), \
		(patch_r_stride*c):(patch_r_stride*c+w_patch)] += y_val_patch_pred_tmp[i]

	return y_val_patch_pred







"""

def make_test_data_multiscale(data_list_, row_size_=16, channel_size_=8, num_patch_=100,\
 isVal_=True, patch_w_stride_=4, patch_c_stride_=2, datadir_='../input/train_data', is_fixed_size=False, \
 med_data_dir_='../input/task2/medInfo.npy'):
	global img_list, label_list, row_size, channel_size, num_patch, patch_dim, patch_w_stride, patch_c_stride
	patch_dim_ = (row_size_ ** 2)*channel_size_

	if isVal_ is True:
		img_list_, label_list_=load_mri_from_directory(data_list_, is_test=False,\
		 datadir='../input/train_data')
		N = len(img_list_)		
		img_list, label_list = img_list_, label_list_	
		row_size, channel_size, num_patch, patch_dim = row_size_, channel_size_, num_patch_, patch_dim_
		patch_w_stride, patch_c_stride = 4, 4

		logging.info("START POOLING: VALIDATION")
		pool = Pool(processes=MULTI)
		result = pool.map(val_core_multiscale, range(N))
		pool.close()
		pool.join()
		logging.info("DONE POOLING: VALIDATION")

		X = []
		X_zoom = []
		Y = []
		GT_list = []
		for j in xrange(len(result)):
			X = X + result[j][0]
			X_zoom = X_zoom + result[j][1]
			Y = Y + result[j][2]
			GT_list = GT_list + result[j][3]

		del img_list, label_list, row_size, channel_size, num_patch, patch_dim, patch_w_stride, patch_c_stride, pool
		
		perm = np.random.permutation(len(Y))
		X, X_zoom, Y, GT_list = np.array(X).astype('float32'), np.array(X_zoom).astype('float32'), np.array(Y), np.array(GT_list)
		X = X[perm]
		X_zoom = X_zoom[perm]    
		Y = Y[perm]
		GT_list = GT_list[perm]

		if addClinical is True:
			medInfo = np.load(med_data_dir_)
			med_list = []
			for j in xrange(N):
				med_list = med_list + [medInfo[data_list_[j]]]*num_patch_
			med_list = np.array(med_list)
			med_list = med_list[perm]
			return X, X_zoom, Y, GT_list, med_list
		else:
			return X, X_zoom, Y, GT_list
	else:
		img_list_ = load_mri_from_directory(data_list_, is_test=(not isVal_), datadir=datadir_, is_fixed_size=is_fixed_size)
		N=len(img_list_)
		img_list = img_list_
		label_list = None
		row_size, channel_size, num_patch, patch_dim = row_size_, channel_size_, num_patch_, patch_dim_
		patch_w_stride, patch_c_stride = patch_w_stride_, patch_c_stride_

		X = []
		X_zoom = []        
		cache = []
		med_list = []
		for k in xrange(N):
			x=[]
			x_zoom=[]            
			for m in xrange(TIME_POINT):
				#Load image
				ex = img_list[k][m]
				ex_zoom = np.ones( ex.shape + np.array([row_size, row_size, channel_size]) )*ex[0,0,0]
				ex_zoom[(row_size/2):(ex.shape[0] + row_size/2), \
				(row_size/2):(ex.shape[1] + row_size/2), \
				(channel_size/2):(ex.shape[2] + channel_size/2)] = ex
				ex_zoom = transform_shrink(ex_zoom)
                #Transpose
				ex = np.transpose(np.array(ex), (2, 0, 1))
				ex_zoom = np.transpose(np.array(ex_zoom), (2, 0, 1))
                #Make patches
				patches = extract_patches(ex, (channel_size, row_size, row_size), \
					extraction_step=(patch_c_stride, patch_w_stride, patch_w_stride))
				x.append(patches)
				patches_zoom = extract_patches(ex_zoom, (channel_size, row_size, row_size), \
					extraction_step=(patch_c_stride/2, patch_w_stride/2, patch_w_stride/2))
				x_zoom.append(patches_zoom)

			N_patches = np.prod(patches.shape) // (channel_size*row_size*row_size)
			X.append(np.array(x))
			X_zoom.append(np.array(x_zoom))            
			cache.append(ex.shape)

			
			
		del img_list, label_list, row_size, channel_size, num_patch, patch_dim, patch_w_stride, patch_c_stride
		if addClinical is True:
			medInfo = np.load(med_data_dir_)
			med_list = med_list + [medInfo[data_list_[k]]]
			return X, X_zoom, cache, med_list
		else:
			return X, X_zoom, cache		
    


def make_train_data_multiscale(data_list_, row_size_=16, channel_size_=8, num_patch_=100, \
	proportion_=0.1):

	img_list_, label_list_=load_mri_from_directory(data_list_, is_test=False, datadir='../input/train_data')
	N=len(label_list_)
	global img_list, label_list, row_size, channel_size, num_patch, proportion
	img_list, label_list = img_list_, label_list_
	row_size, channel_size, num_patch, proportion = row_size_, channel_size_, num_patch_, proportion_
	
	logging.info("START POOLING: TRAINING")
	pool = Pool(processes=MULTI)
	result = pool.map(train_core_multiscale, range(N))
	pool.close()
	pool.join()
	logging.info("DONE POOLING: TRAINING")

	X = []
	X_zoom = []
	Y = []
	GT_list = []
	for j in xrange(len(result)):
		X = X + result[j][0]
		X_zoom = X_zoom + result[j][1]
		Y = Y + result[j][2]
		GT_list = GT_list + result[j][3]

	del img_list, label_list, row_size, channel_size, num_patch, proportion, pool
	X, X_zoom, Y, GT_list = np.array(X).astype('float32'), np.array(X_zoom).astype('float32'), np.array(Y), np.array(GT_list)
	
	if addClinical is True:
		medInfo = np.load('../input/task2/medInfo.npy')
		med_list = []
		for j in xrange(N):
			med_list = med_list + [medInfo[data_list_[j]]]*2*num_patch_
		med_list = np.array(med_list)
		return X, X_zoom, Y, GT_list, med_list
	else:
		return X, X_zoom, Y, GT_list

def train_core_multiscale(k):
	Y = []
	X = []
	X_zoom= []
	GT_list=[]
	label = label_list[k]
	w, _, c =label.shape
  
	m1 = np.where(np.sum(label, axis=1)>0)[0]
	m2 = np.where(np.sum(label, axis=0)>0)[0]
	m3 = np.where(np.sum(label, axis=1)>0)[1]
	patch_dim = (row_size ** 2)*channel_size

	N_GT_patches = int(num_patch*(proportion))
	N_random_patches = num_patch-N_GT_patches
	for ind in zip( np.random.choice(range(make_range(min(m1), w, row_size), \
		make_range(max(m1), w, row_size, False)), N_GT_patches), \
		            np.random.choice(range(make_range(min(m2), w, row_size), \
		                make_range(max(m2), w, row_size, False)), N_GT_patches), \
		            np.random.choice(range(make_range(min(m3), c, channel_size), \
		                make_range(max(m3), c, channel_size, False)), N_GT_patches)):

		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list_zoom = np.array(ind) + np.array(zip((np.arange(8*patch_dim)/(4*row_size*channel_size)), \
		    (np.arange(8*patch_dim)/(2*channel_size))%(2*row_size) , np.arange(8*patch_dim)%(2*channel_size)))

		ind_list = tuple(map(tuple, ind_list))
		ind_list_zoom = tuple(map(tuple, ind_list_zoom))
		y = np.array([(label_list[k])[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		gt = (np.sum(y) > 0)
		x = []
       
		x_zoom = []
		x_saggital = []
		x_saggital_zoom = []
		for m in xrange(TIME_POINT):
			img = np.array([(img_list[k][m])[idx] \
			    for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x.append(img)
			x_saggital.append(transform_saggital(img))
			img_zoom = np.array([(img_list[k][m])[idx] \
			    for idx in ind_list_zoom]).reshape(2*row_size,2*row_size,2*channel_size)
			temp = transform_shrink(img_zoom)
			x_zoom.append(temp)
			x_saggital_zoom.append(transform_saggital(temp))

		Y.append(y)
		X.append(np.array(x))
		X_zoom.append(np.array(x_zoom))
		GT_list.append(gt)

		Y.append(transform_saggital(y))
		X.append(np.array(x_saggital))
		X_zoom.append(np.array(x_saggital_zoom))
		GT_list.append(gt)

	for ind in zip( np.random.choice(w-2*row_size, N_random_patches), \
		                np.random.choice(w-2*row_size, N_random_patches), \
		                np.random.choice(c-2*channel_size, N_random_patches) ) :

		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list_zoom = np.array(ind) + np.array(zip((np.arange(8*patch_dim)/(4*row_size*channel_size)), \
		    (np.arange(8*patch_dim)/(2*channel_size))%(2*row_size) , np.arange(8*patch_dim)%(2*channel_size)))

		ind_list = tuple(map(tuple, ind_list))
		ind_list_zoom = tuple(map(tuple, ind_list_zoom))
		y = np.array([(label_list[k])[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		gt = (np.sum(y) > 0)
		x = []
		x_zoom = []            
		x_saggital = []
		x_saggital_zoom = []
		for m in xrange(TIME_POINT):
			img = np.array([(img_list[k][m])[idx] \
			    for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x.append(img)
			x_saggital.append(transform_saggital(img))
			img_zoom = np.array([(img_list[k][m])[idx] \
			    for idx in ind_list_zoom]).reshape(2*row_size,2*row_size,2*channel_size)
			temp = transform_shrink(img_zoom)
			x_zoom.append(temp)
			x_saggital_zoom.append(transform_saggital(temp))

		Y.append(y)
		X.append(np.array(x))
		X_zoom.append(np.array(x_zoom))
		GT_list.append(gt)
		
		Y.append(transform_saggital(y))
		X.append(np.array(x_saggital))
		X_zoom.append(np.array(x_saggital_zoom))
		GT_list.append(gt)
		
	return X, X_zoom, Y, GT_list

def val_core_multiscale(k):
	Y = []
	X = []
	X_zoom = []  
	GT_list = []      
	w, _, c = img_list[k][0].shape
	for ind in zip( np.random.choice(w-2*row_size, num_patch), \
	                np.random.choice(w-2*row_size, num_patch), \
	                np.random.choice(c-2*channel_size, num_patch)):
		ind_list = np.array(ind) + np.array([row_size/2, row_size/2, channel_size/2]) \
		+ np.array(zip((np.arange(patch_dim)/(row_size*channel_size)), \
		    (np.arange(patch_dim)/channel_size)%row_size , np.arange(patch_dim)%channel_size))

		ind_list_zoom = np.array(ind) + np.array(zip((np.arange(8*patch_dim)/(4*row_size*channel_size)), \
		    (np.arange(8*patch_dim)/(2*channel_size))%(2*row_size) , np.arange(8*patch_dim)%(2*channel_size)))

		ind_list = tuple(map(tuple, ind_list))
		ind_list_zoom = tuple(map(tuple, ind_list_zoom))
		y = np.array([(label_list[k])[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
		gt = (np.sum(y) > 0)
		x = []
		x_zoom = []                
		for m in xrange(TIME_POINT):
			img = np.array([(img_list[k][m])[idx] for idx in ind_list]).reshape(row_size,row_size,channel_size)
			x.append(img)
			img_zoom = np.array([(img_list[k][m])[idx] \
				for idx in ind_list_zoom]).reshape(2*row_size,2*row_size,2*channel_size)
			x_zoom.append(transform_shrink(img_zoom))
		Y.append(y)
		X.append(np.array(x))
		X_zoom.append(np.array(x_zoom))
		GT_list.append(gt)

	return X, X_zoom, Y, GT_list


"""