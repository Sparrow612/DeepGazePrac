import cv2
import tensorflow as tf
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator


class HeadPoseEstimator:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)

        self.image_reviser()  # 截取人脸的正方形区域

        sess = tf.Session()
        self.head_pose_estimator = CnnHeadPoseEstimator(sess)
        self.head_pose_estimator.load_pitch_variables('deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf')
        self.head_pose_estimator.load_yaw_variables('deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf')
        self.head_pose_estimator.load_roll_variables('deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf')

    def image_reviser(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(64, 64)
        )

        print("[INFO] Found {0} Faces.".format(len(faces)))
        (x, y, w, h) = faces[0]
        w = h = max(w, h)
        cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = self.img[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite('face.jpg', face)
        status = cv2.imwrite('faces_detected.jpg', self.img)
        print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
        # self.img = face

    def get_pitch_yaw_roll(self):
        self.img = cv2.resize(src=self.img, dsize=(64, 64))
        pitch = self.head_pose_estimator.return_pitch(self.img)  # 俯仰角
        yaw = self.head_pose_estimator.return_yaw(self.img)  # 偏航角
        roll = self.head_pose_estimator.return_roll(self.img)  # 翻转角
        # 输出结果
        print('Estimated pitch...' + str(pitch[0, 0, 0]) + '度')
        print('Estimated yaw...' + str(yaw[0, 0, 0]) + '度')
        print('Estimated roll...' + str(roll[0, 0, 0]) + '度')

        # 返回字典
        return {'pitch_angle': pitch, 'yaw_angle': yaw, 'roll_angle': roll}


if __name__ == '__main__':
    # get_pitch_yaw_roll('image/csh1.jpg')
    HeadPoseEstimator('face.jpg').get_pitch_yaw_roll()
