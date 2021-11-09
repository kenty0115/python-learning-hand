import mediapipe as mp
import numpy as np
import cv2
import pickle

clf = pickle.load(open('pickle/hand_model.pickle', 'rb'))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.hands

img_par = 1
img_path = "img/background.jpg"
obj_image = cv2.imread(img_path)
# obj_image = cv2.resize(
#     obj_image, (int(obj_image.shape[1] / 2), int(obj_image.shape[0] / 2)))
print(obj_image.shape[1], obj_image.shape[0])


def add_frame(img, fr_pic: int, color: list):
    cp_img = img
    bk1 = np.zeros((fr_pic, cp_img.shape[1], 3), dtype=np.int64)
    bk1[np.where((bk1 == [0, 0, 0]).all(axis=2))] = color
    array = np.insert(cp_img, 0, bk1, axis=0)
    array = np.insert(array, array.shape[0], bk1, axis=0)
    cp_img = array

    bk2 = np.zeros((array.shape[0], fr_pic, 3), dtype=np.int64)
    bk2[np.where((bk2 == [0, 0, 0]).all(axis=2))] = color
    array = np.insert(array, [0], bk2, axis=1)
    array = np.insert(array, [array.shape[1]], bk2, axis=1)
    cp_img = array

    return cp_img


obj_point = [0, 0]
obj_abs = [0, 0]
green_obj = add_frame(obj_image, 5, [0, 255, 0])
red_obj = add_frame(obj_image, 5, [0, 0, 255])

first_loop = True
get_flag = False
cap = cv2.VideoCapture(0)
with mp_holistic.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # cv2.resize(image, (int(image.shape[0] / 3), int(image.shape[1] / 3)))
        wide = image.shape[1]
        high = image.shape[0]

        if first_loop is True:
            first_loop = False
            if obj_image.shape[0]/high > obj_image.shape[1]/wide:
                rsize = high / 2
                obj_image = cv2.resize(
                    obj_image, (int(obj_image.shape[1] * rsize / obj_image.shape[0]), int(rsize)))
            else:
                rsize = wide / 2
                obj_image = cv2.resize(obj_image, (int(rsize), int(
                    obj_image.shape[0] * rsize / obj_image.shape[1])))
            print(obj_image.shape)
            green_obj = add_frame(obj_image, 5, [0, 255, 0])
            red_obj = add_frame(obj_image, 5, [0, 0, 255])
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand_cnt = 0
            for j, hand_landmarks in enumerate(results.multi_hand_landmarks):
                flag = False
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                landmark_list = []
                hand_center = (int((int(wide*hand_landmarks.landmark[0].x)+int(wide*hand_landmarks.landmark[9].x))/2), int(
                    (int(high*hand_landmarks.landmark[0].y)+int(high*hand_landmarks.landmark[9].y))/2))

                for i, landmark in enumerate(hand_landmarks.landmark):
                    x_point = int(wide * landmark.x)
                    y_point = int(high * landmark.y)
                    landmark_list.append(x_point)
                    landmark_list.append(y_point)
                try:
                    pl = clf.predict([landmark_list])
                    hand_num = pl[0]
                    if (obj_point[0] <= hand_center[0] and hand_center[0] <= obj_point[0] + obj_image.shape[1]) and (obj_point[1] <= hand_center[1] and hand_center[1] <= obj_point[1] + obj_image.shape[0]) and hand_cnt == 0:
                        hand_cnt += 1
                        if hand_num == 0 or get_flag is True:
                            if get_flag is False:
                                get_flag = True
                                obj_abs[0] = hand_center[0]
                                obj_abs[1] = hand_center[1]
                            else:
                                obj_point[0] = hand_center[0] - obj_abs[0]
                                obj_point[1] = hand_center[1] - obj_abs[1]
                            print(obj_point)

                            if obj_point[0] < 0:
                                obj_point[0] = 0
                            elif obj_point[0] > wide - obj_image.shape[1]:
                                obj_point[0] = wide - obj_image.shape[1]

                            if obj_point[1] < 0:
                                obj_point[1] = 0
                            elif obj_point[1] > high - obj_image.shape[0]:
                                obj_point[1] = high - obj_image.shape[0]
                            print(obj_point)

                        if get_flag is True and hand_num == 5:
                            get_flag = False

                    cv2.putText(image, str(hand_num), (0, 50*(j+1)),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 5, cv2.LINE_AA)
                except:
                    pass

        if get_flag is True:
            print(obj_point)
            image[obj_point[1]:obj_point[1]+red_obj.shape[0],
                  obj_point[0]:obj_point[0]+red_obj.shape[1]] = red_obj
        else:
            image[obj_point[1]:obj_point[1]+green_obj.shape[0],
                  obj_point[0]:obj_point[0]+green_obj.shape[1]] = green_obj

        # cv2.resize(image_, (int(wide/10), int(high/10)))
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
