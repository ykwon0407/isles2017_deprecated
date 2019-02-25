"""
NEEDS EDIT!!
"""
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import numpy as np
np.random.seed(1004)
from glob import glob
import gc, os, click, sys, time, logging, shutil
import settings, models
from utils import *
from data import *
from medpy.io import load, save

N_ITERS=settings.N_ITERS
N_EPOCHS=settings.N_EPOCHS
N_EPOCHS_FINE=settings.N_EPOCHS_FINE
PATIENCE=settings.PATIENCE
PATIENCE_FINE=settings.PATIENCE_FINE
TIME_POINT=settings.TIME_POINT
ROW_STRIDE=settings.ROW_STRIDE
CHA_STRIDE=settings.CHA_STRIDE
N_REPEAT=settings.N_REPEAT
FIXED_WIDTH=settings.FIXED_WIDTH
FIXED_DEPTH=settings.FIXED_DEPTH


@click.command()
@click.option('--save_path', default='c_2015_test_unet_8_10_13_1_16', show_default=True,
              help="Model configuration files")
def main(save_path):
    start = time.time()

    """
    Header mri
    Flair in 2015, MTT  in 2016, ADC in 2017
    """
    save_path = str(save_path)
    year_of_data = str.split(save_path,'_')[1]
    if year_of_data == '2015':
        mri_vsdnumber='Flair'
        mri_header='Flair'
    else:
        mri_vsdnumber='MTT'
        mri_header='ADC'
        
    # Load configuration
    cnf = '_'.join(str.split(save_path,'_')[:-5])
    CONFIG_DICT=load_module('configs/{}.py'.format(cnf))
    globals().update(CONFIG_DICT)

    # Set logging 
    if os.path.exists('loggings/{}_{}_predict.log'.format(save_path, prediction_dataset)) is True:
        os.remove('loggings/{}_{}_predict.log'.format(save_path, prediction_dataset))
    logging.basicConfig(filename='loggings/{}_{}_predict.log'.format(save_path, prediction_dataset), \
                                level=logging.INFO, stream=sys.stdout)

    stderrLogger=logging.StreamHandler()
    stderrLogger.setFormatter(\
        logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'))
    logging.getLogger().addHandler(stderrLogger)
    logging.info(CONFIG_DICT)
    
    # number of cases
    file_path=test_data_dir+'/*/*/*{}*nii'.format(mri_header)
    N_sample=len(glob(file_path))

    if os.path.exists('submission/{}_{}'.format(save_path, prediction_dataset)) is True:
        shutil.rmtree('submission/{}_{}'.format(save_path, prediction_dataset), ignore_errors=True)
    os.mkdir('submission/{}_{}'.format(save_path, prediction_dataset))
    datadir_to_save='submission/{}_{}/'.format(save_path, prediction_dataset)

    # Cross-validation settings
    logging.info('Creating and compiling model...')
    model_class=find_class_by_name(model_name, models)()
    model=model_class.create_model(channel_size=channel_size, row_size=row_size, \
        n_filter=n_filter, filter_size=filter_size, lr=lr, TIME_POINT=TIME_POINT)

    for seed in xrange(N_REPEAT):
        logging.info('-'*50)
        logging.info("Seed {}".format(seed+1))
        logging.info('-'*50)

        count_folds=0
        kf=KFold(n_splits=5, shuffle=True, random_state=1004+seed)
        for tr_list, te_list in kf.split(np.arange(N_sample)):
            count_folds += 1
            logging.info("Predict {}-Fold".format(count_folds))
            info_check_string='weights/{}/{}_{}.hdf5'.format(save_path, seed, count_folds)
            model.load_weights(info_check_string)

            N_val = len(glob(os.path.join(test_data_dir,'*')))
            loop_list = xrange(N_val)
            subject_list = sorted(glob(os.path.join(test_data_dir,'*')))
            logging.info('test sample size: {}'.format(N_val))

            for i in loop_list:
                X_val_patch, X_val_patch_aug, cache=extract_multiscale_patches_from_mri([i], test_data_dir, \
                                                    is_test=True, is_oversampling=False, row_size=row_size, \
                                                     channel_size=channel_size, num_patch=num_patch, \
                                                     patch_r_stride=row_size/ROW_STRIDE, \
                                                     patch_c_stride=channel_size/CHA_STRIDE, \
                                                    proportion=proportion, is_fixed_size=True, n_time_point=TIME_POINT, \
                                                    fixed_width=FIXED_WIDTH, fixed_depth=FIXED_DEPTH)

                X_val_patch_i = np.transpose(X_val_patch[0].reshape(TIME_POINT, -1, \
                    channel_size, row_size, row_size), (1,0,2,3,4))
                X_val_patch_aug_i = np.transpose(X_val_patch_aug[0].reshape(TIME_POINT, -1, \
                    channel_size, row_size, row_size), (1,0,2,3,4))
                X_val_patch_i, X_val_patch_aug_i = preprocess( \
                        X_val_patch_i, X_val_patch_aug_i, None, 0.0, 1.0, 0.0, 1.0)            

                img_path = sorted(glob(os.path.join(subject_list[i],'*','*{}*.nii'.format(mri_vsdnumber))))[0] 
                logging.info(img_path.split('/')[-1])
                VSDnumber = img_path.split('.')[-2]
                mri_id = str.split(img_path.split('/')[-1], '.')[0]

                img_path = sorted(glob(os.path.join(subject_list[i],'*','*{}*.nii'.format(mri_header))))[0] 
                img_sample, image_header = load(img_path)
                
                N_uncertain = 5 if 'uncertain' in model_name else 1
                y_val_patch=np.transpose(np.array(img_sample), (2,0,1))
                y_val_patch_pred_original_size = np.zeros(y_val_patch.shape)
                save_list=[]
                for k in xrange(N_uncertain):                    
                    y_val_patch_pred_i=model.predict({'main_input':X_val_patch_i, 'aug_input': X_val_patch_aug_i}, \
                        batch_size=batch_size)
                    y_val_patch_pred=make_brain_from_patches(y_val_patch_pred_i, cache[0], patch_r_stride=row_size/ROW_STRIDE, \
                        patch_c_stride=channel_size/CHA_STRIDE)
                    y_val_patch_pred/=((ROW_STRIDE ** 2) * 1.0 * CHA_STRIDE)

                    zoomRate=[float(ai)/bi for ai, bi in zip(y_val_patch.shape, y_val_patch_pred.shape)]
                    # print(cal_dice_coef((transform_shrink(y_val_patch_pred, zoomRate)>0.5).reshape(-1), y_val_patch.reshape(-1)))
                    y_val_patch_pred_original_size+=transform_shrink(y_val_patch_pred, zoomRate)
                    save_list.append(transform_shrink(y_val_patch_pred, zoomRate))
                    del y_val_patch_pred, y_val_patch_pred_i
                    gc.collect()

                
                """
                img_sample = np.transpose(np.array(img_sample), (2,0,1))
                zoomRate = [float(ai)/bi for ai, bi in zip(img_sample.shape, y_val_patch_pred.shape)]
                y_val_patch_pred = transform_shrink(y_val_patch_pred, zoomRate)
                
                # y_val_patch_pred = (y_val_patch_pred > 0.5)
                # y_val_patch_pred = np.array((y_val_patch_pred > 0.5), dtype = np.uint16)
                """

                y_val_patch_pred = np.array(y_val_patch_pred_original_size/N_uncertain, dtype = np.float32)
                y_val_patch_pred = np.transpose(y_val_patch_pred, (1,2,0))

                if prediction_dataset=='train':
                    if i in tr_list:
                        save(y_val_patch_pred, datadir_to_save + \
                            'SMIR.yongyong_{}_{}_{}_tr.{}.nii'.format(prediction_dataset, seed, count_folds, VSDnumber), image_header)
                    else:
                        save(y_val_patch_pred, datadir_to_save + \
                            'SMIR.yongyong_{}_{}_{}_te.{}.nii'.format(prediction_dataset, seed, count_folds, VSDnumber), image_header)
                else:
                    save(y_val_patch_pred, datadir_to_save + \
                            'SMIR.yongyong_{}_{}_{}.{}.nii'.format(prediction_dataset, seed, count_folds, VSDnumber), image_header)

                del X_val_patch_i, X_val_patch_aug_i, X_val_patch

if __name__ == "__main__":
    main()








