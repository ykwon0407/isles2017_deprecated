"""
from glob import glob
import numpy as np
from medpy.io import load, header
from medpy.io import save

def dice(A,B):
    return 2.*np.sum(A*B)/(np.sum(A+B))

N_seed=5
year_of_data='2015'
if year_of_data == '2015':
    mri_vsdnumber='Flair'
    mri_header='Flair'
    data_path='/data/mri/isles/data_2015/ISLES2015_SISS_Testing/*'    
    N_repeat=1
else:
    mri_vsdnumber='MTT'
    mri_header='ADC'
    data_path='../input/train_data_2016/*'
    N_repeat=5

ADC_list=sorted(glob(data_path+'/*{}*'.format(mri_vsdnumber)))
VSD_number_list_all=np.array([str.split(s,'.')[-1] for s in ADC_list])
VSD_number_list_all

result_ensemble=dict()
header_ensemble=dict()
for seed in xrange(N_repeat):
    for j in xrange(5):
        te_list=sorted(glob('../src/submission/c_2015_test_unet_8_10_13_1_16_test/VSD.yongyong_test_{}_{}*'.format(seed, (j+1))))
        te_dict=dict()

        vsd_number_list=[]
        for path in te_list:
            vsd_number = str.split(path,'.')[-2]
            img, img_header=load(path)
            te_dict[vsd_number] = img
            vsd_number_list.append(vsd_number)
            header_ensemble[vsd_number]=img_header

        for vsd_number in vsd_number_list:
            try:
                result_ensemble[vsd_number]+=(te_dict[vsd_number]).astype('float')
            except:
                result_ensemble[vsd_number]=(te_dict[vsd_number]).astype('float')


for vsd_number in VSD_number_list_all:
    mri_header = header_ensemble[vsd_number]
    arr = np.array((result_ensemble[vsd_number]/5. > 0.5) , dtype = np.uint16)
    arr_name='../src/submission/c_2015_test_unet_8_10_13_1_16_test/' + \
        'VSD.final_{}.nii'.format(vsd_number)
    save(arr, arr_name, mri_header)                        
"""

from glob import glob
import os
from medpy.io import load
from medpy.io import save
import numpy as np
from medpy.io import header
import click

@click.command()
@click.option('--interest_list', default='1257', show_default=True,
              help="interested model to ensemble")
@click.option('--train_only', is_flag=True, show_default=True,
              help="ensemble training data only")
@click.option('--proportion', default=0.5, show_default=True,
              help="ensemble proportion")
@click.option('--save_name', default='E1', show_default=True,
              help="interested model to ensemble")
def main(interest_list, train_only, proportion, save_name):
	if train_only == False:
		train_indicator = [False, True]
	else:
		train_indicator = [True]

	for indicator in train_indicator:
		isTrain = indicator

		if isTrain == True:
			datadir_to_load = '../input/submission/pred_result/train/'
			data_name = 'train'
			datadir_to_save = '../input/submission/pred_result_ensemble/train/'
		else:
			datadir_to_load = '../input/submission/pred_result/test/'
			data_name = 'test'
			datadir_to_save = '../input/submission/pred_result_ensemble/test/'

		if os.path.isdir(datadir_to_load) is not True:
			raise
		if os.path.isdir(datadir_to_save) is not True:
			os.makedirs(datadir_to_save)

		file_list = []
		interest_model_list = list(interest_list)
		for model_num in interest_model_list:
			file_list = file_list + glob(os.path.join(datadir_to_load, '*model_{}*.nii'.format(model_num)))
		file_dict = dict()
		for file in file_list:
			name = str.split(file,'.')[-2]
			if name in file_dict.keys():
				file_dict[name].append(file)
			else:
				file_dict[name] = []
				file_dict[name].append(file)
			
		record = []
		for name in file_dict.keys():
			N = len(file_dict[name])
			img_list =[]
			#NEED CHANGE !!!
			if isTrain == True:
				file_path = glob('../input/train_data/*/*/*{}*'.format(name))[0]
			else:
				file_path = glob('../input/test_data/*/*/*{}*'.format(name))[0]
			mtt, mtt_header = load(file_path)
			adc_path = glob('/'.join(str.split(file_path,'/')[:4]) +'/*/*ADC*.nii')[0]
			_, adc_header = load(adc_path)
			# print('name: {}, number of images: {}'.format(name, N))

			for j in xrange(N):
				img, img_header = load(file_dict[name][j])
				if header.get_pixel_spacing(adc_header) != header.get_pixel_spacing(img_header):
					print("ERROR!!!!") 
				if header.get_offset(adc_header) != header.get_offset(img_header):
					print("ERROR!!!!")
				if mtt.shape != img.shape:
					print("ERROR!!!!")
				img_list.append(img)

			arr = np.sum(np.array(img_list), axis=0)
			arr = np.array(arr > N*proportion, dtype = np.uint16)
			arr_name = datadir_to_save + 'VSD.yong_{}_{}.'.format(save_name, data_name) + name + '.nii'
			# print('proportion of GT: {}, dimension: {}'.format(np.mean(arr), arr.shape))
			save(arr, arr_name, adc_header)

			if isTrain == True:
				gt_path = glob('../input/train_data/{}/*/*OT*'.format(str.split(file_path, '/')[-3]))[0]
				gt, gt_header = load(gt_path)
				f1 = 2.0*np.sum(gt*arr)/(np.sum(gt)+np.sum(arr))
				record.append(f1)

		if isTrain == True:
			record = np.array(record)
			print(record)
			print(np.mean(record))
			print(np.std(record))

if __name__ == "__main__":
    main()
