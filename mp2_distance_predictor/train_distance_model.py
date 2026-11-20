import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

def main():
	# ----------- import data and scaling ----------- #
	df = pd.read_csv('../dataset/annotations.csv')
	df = df.dropna()  # Delete frames with no detections
	df = df[np.isfinite(df['distance'])]  # Delete out-of-lane cars

	X = np.array([eval(bbox_str)[0] for bbox_str in df['bbox']])
	y = df[['distance']].values

	# ----------- create model ----------- #	
    # This model takes in a bounding box of a car [[xmin, ymin, xmax, ymax]] and returns the 
    # distance to the car. Simple models will perform well for this problem, but the below 
    # single layer perceptron is too small to perform well. See the keras documentation 
    # https://keras.io/api/layers/ for adding more layers. 
	model = Sequential()
	model.add(Dense(1, input_dim=4, kernel_initializer='normal', activation='relu'))

	# Define your model architecture here

	opt = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='mean_squared_error', optimizer=opt)
	modelname = "distance_model"

	# ----------- define callbacks ----------- #
	earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20,
								  verbose=1, epsilon=1e-4, mode='min')
	
	tensorboard = TensorBoard(log_dir="training_logs/{}".format(modelname))
	checkpoint = ModelCheckpoint(
		filepath=f'{modelname}.h5',
		monitor='val_loss',
		save_best_only=True,
		mode='min'
	)

	# ----------- start training ----------- #
	_ = model.fit(
		X, y,
		validation_split=0.3, epochs=1000, batch_size=256,
		callbacks=[earlyStopping, checkpoint], verbose=1)
	

if __name__ == '__main__':
	main()
