import sys, math
import numpy as np
from voxels import Voxels
from formatting import DataFormatter
from trackml.dataset import load_event

class ModelEvaluator:
    def __init__(self, model):
        self.model = model


    def run(self, hits, x, y):
        voxels = Voxels(hits)
        predicted = self.model.predict(x)
        
        closest_hits = []
        for prediction in predicted:
            x = prediction[0]
            y = prediction[1]
            z = prediction[2]
            closest_hits.append(voxels.findClosestPoint(x, y, z))


# test
if __name__ == "__main__":
    path_to_dataset = "../../Data/train_100_events/"
    event_path = "event000001052"
    hits, cells, particles, truth = load_event(path_to_dataset + event_path)
    
    import tensorflow as tensorflow
    import keras
    from keras.models import load_model

    model = load_model("3in_3out.keras")
    evaluator = ModelEvaluator(model, hits)
