import streamlit as st
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def cal(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def main():
    st.title("Pose Rep Counter")

    Total_count = st.number_input(
        "Enter a number:", min_value=0, max_value=100)

    if st.button("Start Rep Counter"):
        counter = 0
        stage = None
        left_complete = 0
        right_complete = 1

        cap = cv2.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = pose.process(image)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    tmp = results.pose_landmarks.landmark
                    right_wrist = [tmp[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   tmp[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_elbow = [tmp[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   tmp[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_shoulder = [tmp[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      tmp[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    left_wrist = [tmp[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  tmp[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_elbow = [tmp[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  tmp[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_shoulder = [tmp[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     tmp[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                    left_angle = cal(right_shoulder, right_elbow, right_wrist)
                    right_angle = cal(left_shoulder, left_elbow, left_wrist)

                    cv2.putText(image, str(left_angle),
                                tuple(np.multiply(right_elbow, [
                                      width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                    255, 255, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, str(left_angle),
                                tuple(np.multiply(left_elbow, [
                                      width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                    255, 255, 255), 2, cv2.LINE_AA
                                )

                    if left_complete and right_angle > 160:
                        stage = "down"
                    if left_complete and right_angle < 30 and stage == 'down':
                        stage = "up"
                        counter += 1
                        right_complete = 1
                        left_complete = 0

                    if right_complete and left_angle > 160:
                        stage = "down"
                    if right_complete and left_angle < 30 and stage == 'down':
                        stage = "up"
                        counter += 1
                        left_complete = 1
                        right_complete = 0

                except:
                    pass

                cv2.rectangle(image, (0, 0), (275, 73), (245, 117, 16), -1)

                cv2.putText(image, 'REPS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                if counter < Total_count:
                    cv2.putText(image, str(
                        counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Task Completed Press Q For Exit", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
