import numpy as np
import random as rand
from keras import layers
from keras.models import Model, Sequential
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input, LeakyReLU, Activation
from keras.optimizers import Adam, RMSprop
from keras.models import load_model


class Conn4Net(object):
	def __init__(self, input_dim, output_dim, load=False, fname=None):
		self.input_dim = input_dim
		self.output_dim = output_dim

		if load:
			self.load(fname)			
		else:
			self.model = Conn4Net.make_conv_net(input_dim, output_dim)
			#self.model = Conn4Net.make_deep_net(input_dim, output_dim)


	def clone(self):
		net = Conn4Net(self.input_dim, self.output_dim, load=False, fname=None)
		net.model.set_weights(self.model.get_weights())
		return net


	@staticmethod
	def make_deep_net(input_dim, output_dim, lr=0.001):
		input = Input(shape=input_dim)		
		net = Dense(256, activation='relu')(input)
		net = Flatten()(input)
		net = Dense(256, activation='relu')(net)						
		net = BatchNormalization()(net)		

		policy = Dense(256, activation='relu')(net)
		policy = BatchNormalization()(policy)		
		policy = Dense(output_dim)(policy)	
		policy = Activation("softmax")(policy)

		value = Dense(256, activation='relu')(net)
		value = BatchNormalization()(value)		
		value = Dense(1)(value)		
		value = Activation("tanh")(value)
		
		model = Model(inputs=input, outputs=[policy, value])
		model.compile(loss=['categorical_crossentropy', 'mse'], optimizer = Adam(lr=lr))
		return model


	@staticmethod
	def make_conv_net(input_dim, output_dim, lr=0.001):
		input = Input(shape=input_dim)		
		net = Conv2D(256, (3, 3), activation='relu', padding='same')(input)
		#net = MaxPooling2D()(net)
		net = Conv2D(128, (3, 3), activation='relu')(net)
		#net = MaxPooling2D()(net)		
		net = BatchNormalization()(net)		

		policy = Conv2D(256, (1, 1), activation='relu')(net)
		policy = BatchNormalization()(policy)
		policy = Flatten()(policy)
		policy = Dense(output_dim)(policy)		
		policy = Activation("softmax")(policy)

		value = Conv2D(256, (1, 1), activation='relu')(net)
		value = BatchNormalization()(value)	
		value = Flatten()(value)
		value = Dense(1)(value)		
		value = Activation("tanh")(value)
		
		model = Model(inputs=input, outputs=[policy, value])
		model.compile(loss=['categorical_crossentropy', 'mse'], optimizer = Adam(lr=lr))
		return model


	def predict(self, inp):
		pred = self.model.predict(inp)
		return pred[0], pred[1]


	def save(self, fname):
		self.model.save(fname)		


	def load(self, fname):
		self.model = load_model(fname)		


if __name__ == '__main__':	
	from Conn4Game import *
	game = Conn4Game()
	input_dim = game.encoded_board_dim()
	output_dim = game.action_size()
	inp = np.array(game.encode_board()).reshape(1, *input_dim)

	net = Conn4Net(input_dim, output_dim, load=False)	
	pred, v = net.predict(inp)
	print(pred)
	print(v)

