import numpy as np

from keras.models import load_model
from mp2_distance_predictor.detect import detect_cars


class InLanePredictor:
    THRESHOLD = 0.5

    def __init__(self, detection_model_path, in_lane_model_path):
        self.detection_model = load_model(detection_model_path)
        self.in_lane_model = load_model(in_lane_model_path)
    

    def predict(self, image):
        car_bounding_box = detect_cars(self.detection_model, image)  # return the bounding box of car

        in_lane = False
        if car_bounding_box is not None and self.in_lane_model(np.array(car_bounding_box))[0, 0].numpy() > self.THRESHOLD:
            in_lane = True
        return in_lane, car_bounding_box
