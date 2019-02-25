
CONFIG_DICT={
"name" : __name__.split('.')[-1],
"num_patch" : 600, # Number of patches for each case
"proportion" : 0.5, # Oversampling proportion
"row_size" : 24, # Row size
"channel_size" : 4, # Channel size 
"lr" : 1e-4, # learning rate
"batch_size" : 16, # Batch size
"n_filter" : 16, # number of filters
"n_residual" : 0, # Repeat number of residual block
"filter_size" : 3, # filter_size
"data_dir" : '../input/train_data_2017', # input directory
"add_clinical" : False, # indicator whether clinical data is used or not
"finetune" : False, # indicator whether fine-tuning process is conducted [F]
"model_name" : 'shallow_uncertain_unet_test', # model name [M,S]

"N_REPEAT" : 5,
"prediction_dataset" : 'train', # prediction dataset for predict.py
"test_data_dir" : '../input/train_data_2017', # input directory for predict.py
}




