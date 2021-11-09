import mediapipe as mp
import numpy as np
import cv2
import pickle

pickle_path = "pickle"
clf = pickle.load(open(pickle_path + 'hand_model.pickle', 'rb'))
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_holistic.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        # cv2.resize(image, (int(image.shape[0] / 3), int(image.shape[1] / 3)))
        wide = image.shape[1]
        high = image.shape[0]

        if not success:
            print("Ignoring empty camera frame.")
          # ビデオをロードする場合は、「continue」ではなく「break」を使用してください
            continue

        # 後で自分撮りビューを表示するために画像を水平方向に反転し、BGR画像をRGBに変換
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # パフォーマンスを向上させるには、オプションで、参照渡しのためにイメージを書き込み不可としてマーク
        image.flags.writeable = False
        results = hands.process(image)

        # 画像にランドマークアノテーションを描画
        # image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for j, hand_landmarks in enumerate(results.multi_hand_landmarks):
                flag = False
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                landmark_list = []
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x_point = int(wide * landmark.x)
                    y_point = int(high * landmark.y)
                    landmark_list.append(x_point)
                    landmark_list.append(y_point)
                try:
                    pl = clf.predict([landmark_list])
                    hand_num = pl[0]
                    cv2.putText(image, str(hand_num), (0, 50*(j+1)),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 5, cv2.LINE_AA)
                except:
                    pass

        # image = cv2.bitwise_or(image, image_)
        # cv2.resize(image_, (int(wide/10), int(high/10)))
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
