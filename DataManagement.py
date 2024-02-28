import cv2
import mediapipe as mp
import pandas as pd
#import xarray as xr
#import modin.pandas as pd
import time

# https://www.statology.org/pandas-3d-dataframe/

import numpy as np

HAND_REF = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
    'middle_finger_mcp','middle_finger_pip','middle_finger_dip','middle_finger_tip',
    'ring_finger_mcp','ring_finger_pip','ring_finger_dip','ring_finger_tip',
    'pinky_mcp','pinky_pip','pinky_dip','pinky_tip',
]

NUM_POINTS = 21
class DataGuy():

    def __init__(self):
        self.df = None
        self.frames = []

        pass

    def load_from_file(self, path):
        self.df = pd.read_csv(path)
    def from_landmark(self, landmark):
        xarray_3d = xr.Dataset(
            {"product_A": 1}
        )
        d = np.zeros((NUM_POINTS, 3))
        for id, lm in enumerate(landmark):
            d[id][0] = lm.x
            d[id][1] = lm.y
            d[id][2] = lm.z

        df = pd.DataFrame(data=d, columns=['x', 'y', 'z'], index=HAND_REF)
        print(df)
        self.frames.append(df)

    def inc_frame(self):
        self.frame += 1

    def print(self):
        print(self.df)
