import pickle
import numpy as np

def h5_list(data: list):
    pickle_object = pickle.dumps(data)
    return np.void(pickle_object)

def revert_h5_list(data: np.void):
    return pickle.loads(data.tobytes())

