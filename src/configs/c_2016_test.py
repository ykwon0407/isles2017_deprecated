
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
"data_dir" : '../input/train_data_2016', # input directory
"add_clinical" : False, # indicator whether clinical data is used or not
"finetune" : False, # indicator whether fine-tuning process is conducted [F]
"model_name" : 'shallow_unet_2', # model name [M,S]

"prediction_dataset" : 'test', # prediction dataset for predict.py
"test_data_dir" : '../input/test_data_2016', # input directory for predict.py
}




