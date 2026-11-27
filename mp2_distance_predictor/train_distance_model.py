import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Rescaling
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

def main():
	# ----------- import data and filtering ----------- #
	df = pd.read_csv('../dataset/annotations.csv')
	df = df.dropna()  # Delete frames with no detections
	df = df[np.isfinite(df['distance'])]  # Delete out-of-lane cars (inf distances)

	# Parse bounding boxes and ensure valid data
	X = np.array([eval(bbox_str)[0] for bbox_str in df['bbox']])
	y = df[['distance']].values

	# ----------- create model ----------- #	
	# This model takes in a bounding box of a car [xmin, ymin, xmax, ymax] 
	# in 600x600 pixel coordinates and returns the distance to the car in meters.
	# Input normalization is built into the model using a Rescaling layer.
	model = Sequential([
		Rescaling(1.0 / 600.0, input_shape=(4,)),
		Dense(64, activation='relu'),
		Dense(32, activation='relu'),
		Dense(16, activation='relu'),
		Dense(8, activation='relu'),
		Dense(1)  # Output: single distance value (linear activation for regression)
	])

	opt = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
	modelname = "distance_model"

	# ----------- define callbacks ----------- #
	earlyStopping = EarlyStopping(
		monitor='val_loss', 
		patience=15, 
		verbose=1, 
		mode='min',
		restore_best_weights=True
	)
	reduce_lr = ReduceLROnPlateau(
		monitor='val_loss', 
		factor=0.8, 
		patience=10,
		verbose=1, 
		epsilon=1e-4, 
		mode='min'
	)
	
	tensorboard = TensorBoard(log_dir="training_logs/{}".format(modelname))
	checkpoint = ModelCheckpoint(
		filepath=f'{modelname}.h5',
		monitor='val_loss',
		save_best_only=True,
		mode='min'
	)

	# ----------- start training ----------- #
	print(f"Training on {len(X)} samples with 80/20 train/validation split")
	history = model.fit(
		X, y,
		validation_split=0.2,
		epochs=1000,
		batch_size=32,
		callbacks=[earlyStopping, reduce_lr, checkpoint],
		verbose=1)

	# ----------- final evaluation ----------- #
	print("\n" + "="*50)
	print("Training completed. Final model saved to distance_model.h5")
	print("="*50)
	

if __name__ == '__main__':
	main()
