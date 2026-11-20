import numpy as np

from keras.models import load_model


class DistancePredictor:
    def __init__(self, distance_model_path):
        self.distance_model = load_model(distance_model_path)

    def predict(self, car_bounding_box) -> float:
        dist = float('inf')
        if car_bounding_box is not None:
            dist = self.distance_model(np.array(car_bounding_box))[0, 0].numpy()
        return dist
