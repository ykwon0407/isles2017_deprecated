#!/bin/bash

echo "Heroes Never Die!"

# THEANO_FLAGS='floatX=float32,device=cuda0,lib.cnmem=0.95' python predict.py --save_path c_2015_test_unet_8_10_13_1_16
# THEANO_FLAGS='floatX=float32,device=cuda0,lib.cnmem=0.95' python predict.py --save_path c_2016_test_8_8_14_16_6


# THEANO_FLAGS='floatX=float32,device=cuda3,lib.cnmem=0.95' python train_multi.py --cnf c_2017_test_unet

# THEANO_FLAGS='floatX=float32,device=cuda3,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test
# THEANO_FLAGS='floatX=float32,device=cuda2,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test_unet
# THEANO_FLAGS='floatX=float32,device=cuda0,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test_shallow
# THEANO_FLAGS='floatX=float32,device=cuda2,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test_fine
# THEANO_FLAGS='floatX=float32,device=cuda1,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test_shallow_2



# THEANO_FLAGS='floatX=float32,device=cuda3,lib.cnmem=0.95' python train_multi.py --cnf c_2015_test_shallow
# THEANO_FLAGS='floatX=float32,device=cuda3,lib.cnmem=0.95' python train_multi.py --cnf c_2015_test_unet

# THEANO_FLAGS='floatX=float32,device=cuda2,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test_2
# THEANO_FLAGS='floatX=float32,device=cuda1,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test_3
# THEANO_FLAGS='floatX=float32,device=cuda0,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test_unet_2

# THEANO_FLAGS='floatX=float32,device=cuda3,lib.cnmem=0.95' python train_multi.py --cnf c_2015_test
# THEANO_FLAGS='floatX=float32,device=cuda2,lib.cnmem=0.95' python train_multi.py --cnf c_2016_test_shallow
# THEANO_FLAGS='floatX=float32,device=cuda2,lib.cnmem=0.95' python train_multi.py --cnf c_2017_uncertain



# THEANO_FLAGS='floatX=float32,device=cuda2,lib.cnmem=0.95' python train_multi.py --cnf c_2017_uncertain

# THEANO_FLAGS='floatX=float32,device=cuda3,lib.cnmem=0.95' python predict.py --save_path c_2017_uncertain_2_8_20_20_55_10
# THEANO_FLAGS='floatX=float32,device=cuda0,lib.cnmem=0.95' python predict.py --save_path c_2017_uncertain_2_8_25_17_27_22
# THEANO_FLAGS='floatX=float32,device=cuda1,lib.cnmem=0.95' python predict.py --save_path c_2017_uncertain_8_21_11_14_5


# THEANO_FLAGS='floatX=float32,device=cuda3,lib.cnmem=0.95' python train_multi.py --cnf c_2017_uncertain_nopool
