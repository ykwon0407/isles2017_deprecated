from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import numpy as np
import pandas as pd
np.random.seed(1004)
from glob import glob
import gc, os, click, sys, time, logging, shutil
import settings, models
from utils import *
from data import *

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
@click.option('--cnf', default='c_single_model', show_default=True,
              help="Model configuration files")
def main(cnf):
    start = time.time()
    # Load configuration
    CONFIG_DICT=load_module('configs/{}.py'.format(cnf))
    globals().update(CONFIG_DICT)

    # Set logging 
    if os.path.exists('loggings/{}.log'.format(name)) is True:
        os.remove('loggings/{}.log'.format(name))
    logging.basicConfig(filename='loggings/{}.log'.format(name), \
                                level=logging.INFO, stream=sys.stdout)
    stderrLogger=logging.StreamHandler()
    stderrLogger.setFormatter(\
        logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'))
    logging.getLogger().addHandler(stderrLogger)
    logging.info(CONFIG_DICT)

    # Cross-validation settings
    file_path=data_dir+'/*/*/*OT*nii'
    N_sample=len(glob(file_path))

    if os.path.exists('weights/{}'.format(name)) is True:
        shutil.rmtree('weights/{}'.format(name), ignore_errors=True)
    os.mkdir('weights/{}'.format(name))
    
    result=np.zeros((N_sample, N_REPEAT))
    for seed in xrange(N_REPEAT):
        logging.info('-'*50)
        logging.info("Seed {}".format(seed+1))
        logging.info('-'*50)

        count_folds=0
        kf=KFold(n_splits=5, shuffle=True, random_state=1004+seed)
        for tr_list, te_list in kf.split(np.arange(N_sample)):
            count_folds += 1
            logging.info('-'*50)
            logging.info("Train {}-Fold".format(count_folds))
            logging.info('-'*50)
            
            logging.info("Load Train")
            X_train, y_train, lesion_indicator_train=extract_patches_from_mri(tr_list, data_dir, \
                                                        is_test=False, is_oversampling=True, row_size=row_size, \
                                                         channel_size=channel_size, num_patch=num_patch, \
                                                            proportion=proportion, \
                                                            fixed_width=FIXED_WIDTH, fixed_depth=FIXED_DEPTH)

            X_train, y_train, mean, std=preprocess(X_train, y_train, None, None)

            logging.info("Load Validation")
            X_val, y_val, lesion_indicator_val=extract_patches_from_mri(te_list, data_dir, \
                                                        is_test=False, is_oversampling=False, row_size=row_size, \
                                                         channel_size=channel_size, num_patch=num_patch, \
                                                            proportion=proportion, \
                                                            fixed_width=FIXED_WIDTH, fixed_depth=FIXED_DEPTH)
            X_val, y_val = preprocess(X_val, y_val, mean, std)

            logging.info("Load Model")
            model_class=find_class_by_name(model_name, models)()
            model=model_class.create_model(channel_size=channel_size, row_size=row_size, \
                n_filter=n_filter, filter_size=filter_size, lr=lr, TIME_POINT=TIME_POINT)
            
            logging.info('-'*50)
            logging.info('Fitting : compile.....')
            logging.info('-'*50)
            
            # Callbacks
            SCHEDULER=lambda epoch:lr*(0.99 ** epoch)
            info_check_string='weights/{}/{}_{}.hdf5'.format(name, seed, count_folds)
            early_stopping=EarlyStopping(monitor='val_loss', patience=PATIENCE)
            model_checkpoint=ModelCheckpoint(info_check_string, monitor='loss', save_best_only=True)
            change_lr=LearningRateScheduler(SCHEDULER) 

            b_generator = balance_generator(X_train, y_train, lesion_indicator_train, batch_size)
            model.fit_generator(b_generator, steps_per_epoch=N_ITERS, epochs=N_EPOCHS, \
             validation_data=({'main_input':X_val},\
              y_val), callbacks=[early_stopping, model_checkpoint, change_lr])

            model.load_weights(info_check_string)
            
            if finetune is True:
                logging.info('-'*50)
                logging.info('Finetuning')
                logging.info('-'*50)
                # Data load
                X_train, y_train, lesion_indicator_train=extract_patches_from_mri(tr_list, data_dir, \
                                                            is_test=False, is_oversampling=False, row_size=row_size, \
                                                             channel_size=channel_size, num_patch=num_patch, \
                                                                proportion=proportion, \
                                                            fixed_width=FIXED_WIDTH, fixed_depth=FIXED_DEPTH)
                X_train, y_train=preprocess(X_train, y_train, mean, std)
                
                # Callbacks
                SCHEDULER_FINE=lambda epoch:lr*(0.99 ** epoch)/15.0
                info_check_string_fine='weights/{}/fine_{}_{}.hdf5'.format(name, seed, count_folds)
                early_stopping_fine=EarlyStopping(monitor='val_loss', patience=PATIENCE)
                model_checkpoint_fine=ModelCheckpoint(info_check_string_fine, monitor='loss', save_best_only=True)
                change_lr_fine=LearningRateScheduler(SCHEDULER_FINE)

                b_generator_fine=balance_generator(X_train, y_train, lesion_indicator_train, batch_size)
                model.fit_generator(b_generator_fine, steps_per_epoch=N_ITERS, epochs=N_EPOCHS_FINE, \
                 validation_data=({'main_input':X_val},\
                  y_val), callbacks=[early_stopping_fine, model_checkpoint_fine, change_lr_fine])

                model.load_weights(info_check_string_fine)

            logging.info('-'*50)
            logging.info('Validating')
            logging.info('-'*50)
            
            _, label_list=load_mri_from_directory(te_list, , FIXED_WIDTH, FIXED_DEPTH,\
                                        is_test=False, data_dir=data_dir, is_fixed_size=False)
            X_val_patch, cache=extract_patches_from_mri(te_list, data_dir, \
                                                is_test=True, is_oversampling=False, row_size=row_size, \
                                                 channel_size=channel_size, num_patch=num_patch, \
                                                 patch_r_stride=row_size/ROW_STRIDE, \
                                                 patch_c_stride=channel_size/CHA_STRIDE, \
                                                proportion=proportion, is_fixed_size=True, \
                                                fixed_width=FIXED_WIDTH, fixed_depth=FIXED_DEPTH)

            N_val = len(X_val_patch)   
            for i in xrange(N_val):
                list_sum_of_GT_by_depth_axis=[]
                X_val_patch_i=np.transpose(X_val_patch[i].reshape(TIME_POINT, -1, \
                    channel_size, row_size, row_size), (1,0,2,3,4) )
                X_val_patch_i = preprocess(X_val_patch_i, None, mean, std)
                y_val_patch_pred_i=model.predict({'main_input':X_val_patch_i}, \
                    batch_size=batch_size)
                y_val_patch_pred=make_brain_from_patches(y_val_patch_pred_i, cache[i], patch_r_stride=row_size/ROW_STRIDE, \
                    patch_c_stride=channel_size/CHA_STRIDE)

                y_val_patch_pred/=((ROW_STRIDE ** 2) * 1.0 * CHA_STRIDE)

                y_val_patch=np.transpose(np.array(label_list[i]), (2,0,1))
                zoomRate=[float(ai)/bi for ai, bi in zip(y_val_patch.shape, y_val_patch_pred.shape)]
                y_val_patch_pred=transform_shrink(y_val_patch_pred, zoomRate)
                y_val_patch_pred=(y_val_patch_pred > 0.5)
                
                logging.info('data:{}, pred: {}, GT: {}'.format( (i+1), np.mean(y_val_patch_pred), np.mean(y_val_patch)))
                for j in xrange(y_val_patch_pred.shape[0]):
                    list_sum_of_GT_by_depth_axis.append([np.sum(y_val_patch_pred[j]), np.sum(y_val_patch[j])])
                logging.info(list_sum_of_GT_by_depth_axis)
                
                dice_coef=cal_dice_coef(y_val_patch_pred.reshape(-1), y_val_patch.reshape(-1))            
                logging.info('Dice Coef: {}'.format(dice_coef))
                result[te_list[i], seed]=dice_coef

                del X_val_patch_i
                del y_val_patch_pred, y_val_patch_pred_i
                gc.collect()

            logging.info("Number of parameters: {}".format(model.count_params()))
            del X_train, y_train
            del X_val, y_val
            del X_val_patch
            del model
            gc.collect()

        logging.info("RESULT: \n {}".format(result[:,seed]))
        logging.info("MEAN: {}".format(np.mean(result[:,seed])))
        logging.info("STD: {}".format(np.std(result[:,seed])))
    
    logging.info("-"*50)
    logging.info("RESULT")
    logging.info("-"*50)
    logging.info("MEAN: {}".format(np.mean(result)))
    logging.info("STD: {}".format(np.std(result)))

    pd.DataFrame(result).to_csv('weights/{}/result.csv'.format(name), index=False)
    
    end = time.time()
    logging.info("Elapsed time: {}".format(end-start))
    logging.info(CONFIG_DICT)    
    
if __name__ == "__main__":
    main()


