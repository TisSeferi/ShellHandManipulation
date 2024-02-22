import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import DataManagement as dm

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
cTime = 0

NUM_POINTS = 21
HAND_REF = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
    'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
    'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

DataGuy = dm.DataGuy()


def to_data_frame(landmark):
    d = np.zeros((NUM_POINTS, 3))
    for id, lm in enumerate(landmark):
        d[id][0] = lm.x
        d[id][1] = lm.y
        d[id][2] = lm.z

    df = pd.DataFrame(data=d, columns=['x', 'y', 'z'], index=HAND_REF)
    return df


def Euclidean_Dist(df1, df2, cols=['x', 'y', 'z']):
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)


def centroid(df, centr_selection=None):
    if centr_selection is None:
        #centr_selection = [
        #    'wrist',  # 0
        #    'thumb_ip',
        #    'thumb_tip',
        #    'index_finger_mcp',
        #    'pinky_mcp'
        #]
        centr_selection = [
            'wrist',  # 0
            'thumb_ip',  # 2,
            'index_finger_pip',  # 6,
            'middle_finger_pip',#10,
            'ring_finger_pip',#14,
            'pinky_pip',  # 18
        ]
        # centr_selection = [
        #    'wrist',#0
        #    'thumb_mcp',#2,
        #    'index_finger_mcp',#6,
        #    #'middle_finger_mcp',#10,
        #    #'ring_finger_mcp',#14,
        #    'pinky_mcp',#18
        # ]
        # centr_selection = [
        #    'wrist',
        #    'thumb_tip',
        #    'index_finger_tip',
        #    'middle_finger_tip'
        #    'ring_finger_tip',
        #    'pinky_tip'
        # ]
    circ = df.loc[centr_selection]
    return circ.mean()

    #eturn df.loc['wrist']


def fings_up(df):
    tip_selection = [
        'thumb_tip',  # 4
        'index_finger_tip',  # 8,
        'middle_finger_tip',  # 12,
        'ring_finger_tip',  # 16,
        'pinky_tip',  # 18
    ]

    knuck_selection = [
        'thumb_mcp',  # 2,
        'index_finger_pip',  # 6,
        'middle_finger_pip',  # 10,
        'ring_finger_pip',  # 14,
        'pinky_pip',  # 18
    ]
    row_names = [
        'thumb',
        'index_finger',
        'middle_finger',
        'ring_finger',
        'pinky',
    ]
    cols = ['x','y','z']
    wrist = df.loc['wrist']
    thumb_ref = df.loc['index_finger_pip']

    knucks = Euclidean_Dist(df.loc[knuck_selection], wrist)
    knucks[0] = Euclidean_Dist(df.loc[knuck_selection], thumb_ref)[0] * .8 # thumb correction to index_finger_mcp
    tips = Euclidean_Dist(df.loc[tip_selection], wrist)
    tips[0] = Euclidean_Dist(df.loc[tip_selection], thumb_ref)[0]
    ret = np.greater_equal(tips, knucks)

    return ret


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # DataGuy.from_landmark(handLms.landmark)
            temp = to_data_frame(handLms.landmark)
            fingos = fings_up(temp)

            row_names = [
                'thumb',
                'index_finger',
                'middle_finger',
                'ring_finger',
                'pinky',
            ]
            h, w, c = img.shape

            a = temp.loc['pinky_mcp']
            b = temp.loc['index_finger_mcp']

            if a['x'] > b['x']:
                ref = a
            else:
                ref = b

            x_offset = 20
            start_x = x_offset + int(ref['x'] * w)
            start_y = int(ref['y'] * h)

            #start_x = 10
            end_x = start_x + 170
            #start_y = 300
            dy = 30
            cv2.rectangle(img, (start_x - 20, start_y - dy), (end_x, start_y + 5 * dy), (0, 0, 0), -1)
            for ind, val in enumerate(row_names):
                text = val + ': ' + ('Up' if fingos[ind] else 'Down')
                cv2.putText(img, text, (start_x, start_y + ind * dy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            # cv2.putText(img, disp, (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
