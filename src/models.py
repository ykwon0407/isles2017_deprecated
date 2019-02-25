from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from keras.layers.merge import concatenate, add
from keras.layers import Dense, Activation, ELU, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from theano import tensor as T
from keras import regularizers
from keras.engine.topology import Layer
import settings
from keras.legacy import interfaces

DROP_RATE=settings.DROP_RATE
EPSILON=settings.EPSILON

class Dropout_uncertain(Layer):
    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout_uncertain, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, _):
        return self.noise_shape

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=True)
        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(Dropout_uncertain, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

class SampleNormal(Layer):
    __name__ = 'sample_normal'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(SampleNormal, self).__init__(**kwargs)

    def _sample_normal(self, z_avg, z_log_var):
        eps = K.random_normal(shape=K.shape(z_avg), mean=0.0, stddev=0.2) #0.2 TODO
        # eps = K.random_uniform(shape=K.shape(z_avg), \
        #  minval=-1./1.73205, maxval=1./1.73205)
        return z_avg + K.exp(z_log_var)*eps

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        return self._sample_normal(z_avg, z_log_var)


def dice_coef(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = 2. * K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) + 1e-8
    return (intersection / union) 

def recall_smooth(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection / (K.sum(y_true_f)+ 1e-8))    

def dice_coef_loss(y_true, y_pred):
	"""
	for smoothed dice_coef_loss one can use
	y_true_f = K.clip(K.flatten(y_true), EPSILON, 1-EPSILON)
	"""
	y_pred_f = K.flatten(y_pred)
	y_true_f = K.flatten(y_true) 
	intersection = 2. * K.sum(y_true_f * y_pred_f)
	union = K.sum(y_true_f) + K.sum(y_pred_f) + 1e-8
	return -(intersection / union) 

def _basic_block(n=16, c=1, w=1, h=1):
    def f(input):
        conv_output1 = Conv3D(n, (c, w, h), activation='linear', padding='same', kernel_initializer='he_normal')(input)
        act_output1 = ELU(1.0)(conv_output1)
        conv_output2 = Conv3D(n, (c, w, h), activation='linear', padding='same', kernel_initializer='he_normal')(act_output1)
        return BatchNormalization(axis=1)(conv_output2)
    return f     

def _residual_block(n=16, c=1, w=1, h=1, activation='linear', padding='same'):
    def f(input):
        residual = _basic_block(n, c, w, h)(input)
        return add([input, residual])
    return f

def S_Convolution3D(n=16, c=1, w=1, h=1, activation='linear', padding='same', strides=(1, 1, 1)):
	def f(input):
		conv = BatchNormalization(axis=1)(input)
		conv = Conv3D(n, (c, w, h), strides=strides, \
			activation=activation, padding=padding, \
			kernel_initializer='he_normal')(conv)
		return ELU(1.0)(conv)
	return f

def NConvolution3D(n=16, c=1, w=1, h=1, activation='linear', padding='same'):
    def f(input):
        conv = Conv3D(n, (c, w, h), activation=activation, padding=padding, kernel_initializer='he_normal')(input)
        act_output = ELU(1.0)(conv)
        return BatchNormalization(axis=1)(act_output)
    return f   

def conv_bn_relu(n=16, c=1, w=1, h=1, activation='linear', padding='same'):
	"""
	Not really good
	"""
	def f(input):
		conv = Conv3D(n, (c, w, h), activation=activation, padding=padding, kernel_initializer='he_normal')(input)
		bn = BatchNormalization(axis=1)(conv)
		return ELU(1.0)(bn)
	return f   

class base(object):

	def create_model(self, unused_model_input, **unused_params):
		raise NotImplementedError()


class shallowest_unet_3(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = Dropout(DROP_RATE)(conv2)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_conv2)

		#Merge two parts
		up3 = concatenate([UpSampling3D(size=(2, 2, 2))(pool2), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_pool2), scaled_conv1], axis=1)
		conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(up3)
		conv3 = Dropout(DROP_RATE)(conv3)

		conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(conv3)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Softmax
		conv5 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv4)
		model = Model(inputs=[main_input, scaled_input], outputs=conv5)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallowest_unet_2(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = Dropout(DROP_RATE)(conv2)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_conv2)

		#Merge two parts
		up3 = concatenate([UpSampling3D(size=(2, 2, 2))(pool2), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_pool2), scaled_conv1], axis=1)
		conv3 = NConvolution3D(4*n_filter, 3, 3, 3)(up3)
		conv3 = Dropout(DROP_RATE)(conv3)

		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(conv3)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Softmax
		conv5 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv4)
		model = Model(inputs=[main_input, scaled_input], outputs=conv5)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallowest_unet(base):
	"""
	Not much different with shallowest_unet_2
	"""
	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = Dropout(DROP_RATE)(conv2)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_conv2)

		#Merge two parts
		up3 = concatenate([UpSampling3D(size=(2, 2, 2))(pool2), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_pool2), scaled_conv1], axis=1)
		conv3 = NConvolution3D(4*n_filter, 3, 3, 3)(up3)  # TODO this makes many parameters
		conv3 = Dropout(DROP_RATE)(conv3)

		conv4 = NConvolution3D(2*n_filter, 1, 3, 1)(conv3)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Softmax
		conv5 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv4)
		model = Model(inputs=[main_input, scaled_input], outputs=conv5)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallow_unet(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(4*n_filter, 1, 1, 1)(up5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallow_unet_2_aleatoric(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(pool2)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(up4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(scaled_pool2)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_up4)
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5) 
		conv5 = NConvolution3D(3*n_filter, 1, 3, 3)(conv5)
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		# sampling normal
		z_avg=Conv3D(1, (1, 3, 3), padding='same', activation='linear')(conv6)
		z_log_var=Conv3D(1, (1, 3, 3), padding='same', activation='linear')(conv6)

		z=SampleNormal()([z_avg, z_log_var])
		conv7=Activation('sigmoid')(z)

		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallow_uncertain_unet_test(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(4*n_filter, 1, 1, 1)(up5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class uncertain_unet(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5) 
		conv5 = NConvolution3D(3*n_filter, 1, 3, 3)(conv5)
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		# sampling normal
		z_avg=Conv3D(1, (1, 3, 3), \
			kernel_regularizer=regularizers.l2(1e-8), \
			padding='same', activation='linear')(conv6)
		z_log_var=Conv3D(1, (1, 3, 3), \
			kernel_regularizer=regularizers.l2(1e-8), \
			padding='same', activation='linear')(conv6)

		z=SampleNormal()([z_avg, z_log_var])
		conv7=Activation('sigmoid')(z)

		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model



class cate_uncertain_shallow_unet(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class cate_uncertain_shallow_unet_4(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 3, 3)(up5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model		

class cate_uncertain_shallow_unet_3(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(4*n_filter, 1, 1, 1)(up5)
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class cate_uncertain_test(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		conv1 = Dropout_uncertain(DROP_RATE)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		conv2 = Dropout_uncertain(DROP_RATE)(conv2)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_conv1 = Dropout_uncertain(DROP_RATE)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		
		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_pool1)
		scaled_conv2 = Dropout_uncertain(DROP_RATE)(scaled_conv2)

		#Merge two parts
		up3 = concatenate([UpSampling3D(size=(2, 2, 2))(conv2), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv2), scaled_conv1], axis=1)
		conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(up3)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)
		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(conv3)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Softmax
		conv5 = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid')(conv4)
		model = Model(inputs=[main_input, scaled_input], outputs=conv5)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model				

class cate_uncertain_shallow_unet_2(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class cate_uncertain_unet(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5) 
		conv5 = NConvolution3D(3*n_filter, 1, 3, 3)(conv5)
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid')(conv6)

		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model



class shallow_unet_2(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(pool2)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(up4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 1, 3, 3)(scaled_pool2)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_up4)
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5) 
		conv5 = NConvolution3D(3*n_filter, 1, 3, 3)(conv5)
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class unet(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, filter_size, filter_size)(conv2)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 1, filter_size, filter_size)(conv4) 
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_conv2 = NConvolution3D(2*n_filter, 1, filter_size, filter_size)(scaled_conv2)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(scaled_conv3)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, filter_size, filter_size)(scaled_conv4) 
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(4*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(4*n_filter, 1, 3, 3)(conv5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model


class unet_2(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 1, 1)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(conv2)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(pool2)
		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(conv4) 
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_conv1 = NConvolution3D(n_filter, 1, 1, 1)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_conv2)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_conv3)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_conv4) 
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(2*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallowest_unet_2015(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		pool2 = Dropout(DROP_RATE)(conv2)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_conv2)

		#Merge two parts
		up3 = concatenate([UpSampling3D(size=(2, 2, 2))(pool2), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_pool2), scaled_conv1], axis=1)
		conv3 = NConvolution3D(4*n_filter, 3, 3, 3)(up3)
		conv3 = Dropout(DROP_RATE)(conv3)

		conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(conv3)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Softmax
		conv5 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv4)
		model = Model(inputs=[main_input, scaled_input], outputs=conv5)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallow_unet_2015(base):
	"""
	Might be too shallow....
	"""
	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 3, 3, 3)(pool2)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(4*n_filter, 3, 3, 3)(scaled_pool2)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(4*n_filter, 3, 3, 3)(up5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 3, 3, 3)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallow_unet_2015_2(base):
	"""
	Might be too shallow....
	"""
	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(2*n_filter, 3, 3, 3)(up5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model


class unet_2015(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 1, 1)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(conv2)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(pool2)
		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(conv4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_conv1 = NConvolution3D(n_filter, 1, 1, 1)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_conv2)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(scaled_pool2)
		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_conv3)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_conv4)
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(2*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(2*n_filter, 3, 3, 3)(conv5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model		

class unet_2015_2(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 1, 1)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(conv2)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(pool2)
		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(conv4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_conv1 = NConvolution3D(n_filter, 1, 1, 1)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_conv2)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(scaled_pool2)
		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_conv3)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_conv4)
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(3*n_filter, 3, 3, 3)(conv5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 3, 3, 3)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model		


class cate_uncertain_shallow_unet_2015(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 3, 3, 3)(up5)
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 3, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model


class unet_cate_uncertain_2015(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 1, 1)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(conv2)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(pool2)
		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(conv4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_conv1 = NConvolution3D(n_filter, 1, 1, 1)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_conv2)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(scaled_pool2)
		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_conv3)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_conv4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(3*n_filter, 3, 3, 3)(conv5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 3, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model


class unet_uncertain_2015(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 1, 1)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(conv2)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(pool2)
		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(conv4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_conv1 = NConvolution3D(n_filter, 1, 1, 1)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_conv2)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(scaled_pool2)
		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_conv3)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_conv4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(3*n_filter, 3, 3, 3)(conv5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 3, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		# sampling normal
		z_avg=Conv3D(1, (1, 3, 3), \
			kernel_regularizer=regularizers.l2(1e-8), \
			padding='same', activation='linear')(conv6)
		z_log_var=Conv3D(1, (1, 3, 3), \
			kernel_regularizer=regularizers.l2(1e-8), \
			padding='same', activation='linear')(conv6)

		z=SampleNormal()([z_avg, z_log_var])
		conv7=Activation('sigmoid')(z)

		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model	


class shallow_unet_pooldrop(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = S_Convolution3D(n_filter, 3, 3, 3)(main_input)
		pool1 = S_Convolution3D(n_filter, 1, 1, 1, strides=(2,2,2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = S_Convolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = S_Convolution3D(2*n_filter, 1, 1, 1, strides=(1,2,2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = S_Convolution3D(3*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = S_Convolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)


		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = S_Convolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_pool1 = S_Convolution3D(n_filter, 1, 1, 1, strides=(2,2,2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = S_Convolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = S_Convolution3D(2*n_filter, 1, 1, 1, strides=(1,2,2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = S_Convolution3D(3*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = S_Convolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = S_Convolution3D(3*n_filter, 1, 3, 3)(up5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = S_Convolution3D(2*n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallow_unet_noMax(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = S_Convolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = S_Convolution3D(n_filter, 1, 3, 3, strides=(2,2,2))(conv1)

		conv2 = S_Convolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = S_Convolution3D(2*n_filter, 1, 3, 3, strides=(1,2,2))(conv2)

		conv3 = S_Convolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = S_Convolution3D(4*n_filter, 1, 1, 1)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = S_Convolution3D(2*n_filter, 1, 1, 1)(up4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = S_Convolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = S_Convolution3D(n_filter, 1, 3, 3, strides=(2,2,2))(scaled_conv1)

		scaled_conv2 = S_Convolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = S_Convolution3D(2*n_filter, 1, 3, 3, strides=(1,2,2))(scaled_conv2)

		scaled_conv3 = S_Convolution3D(4*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = S_Convolution3D(4*n_filter, 1, 1, 1)(scaled_pool2)

		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_conv4 = S_Convolution3D(2*n_filter, 1, 1, 1)(scaled_up4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = S_Convolution3D(4*n_filter, 1, 3, 3)(up5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = S_Convolution3D(2*n_filter, 1, 3, 3)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model		



"""
class shallow_unet_single(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1], axis=1)
		conv5 = NConvolution3D(4*n_filter, 1, 1, 1)(up5)
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=main_input, outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class unet_single(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, filter_size, filter_size)(conv2)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 1, filter_size, filter_size)(conv4) # samll
		conv4 = Dropout(DROP_RATE)(conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1], axis=1)
		conv5 = NConvolution3D(4*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(2*n_filter, 1, 3, 3)(conv5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=main_input, outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model

class shallow_unet_residual(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 1, 3, 3)(main_input)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(pool1)
		pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
		pool2 = Dropout(DROP_RATE)(pool2)

		conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(pool2)
		conv3 = Dropout(DROP_RATE)(conv3)

		conv2_resi = _residual_block(2*n_filter, 1, 3, 3, activation='linear', padding='same')(conv2) 
		up4 = concatenate([UpSampling3D(size=(1, 2, 2))(conv3), conv2_resi], axis=1)
		conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = Dropout(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 1, 3, 3)(scaled_input)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 1, 3, 3)(scaled_pool1)
		scaled_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(4*n_filter, 1, 1, 1)(scaled_pool2)
		scaled_conv3 = Dropout(DROP_RATE)(scaled_conv3)  

		scaled_conv2_resi = _residual_block(2*n_filter, 1, 3, 3, activation='linear', padding='same')(scaled_conv2) 
		scaled_up4 = concatenate([UpSampling3D(size=(1, 2, 2))(scaled_conv3), scaled_conv2_resi], axis=1)
		scaled_conv4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = Dropout(DROP_RATE)(scaled_conv4)

		#Merge two parts
		conv1_resi = _residual_block(n_filter, 1, 3, 3, activation='linear', padding='same')(conv1) 
		scaled_conv1_resi = _residual_block(n_filter, 1, 3, 3, activation='linear', padding='same')(scaled_conv1) 
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1_resi, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1_resi], axis=1)
		conv5 = NConvolution3D(4*n_filter, 1, 1, 1)(up5) 
		conv5 = Dropout(DROP_RATE)(conv5)

		conv6 = NConvolution3D(n_filter, 1, 1, 1)(conv5)
		conv6 = Dropout(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model
"""
