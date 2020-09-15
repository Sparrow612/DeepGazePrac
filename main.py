import cv2
import tensorflow as tf
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator


def get_pitch_yaw_roll(img):
    sess = tf.Session()
    head_pose_estimator = CnnHeadPoseEstimator(sess)

    head_pose_estimator.load_pitch_variables('deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf')
    head_pose_estimator.load_yaw_variables('deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf')
    head_pose_estimator.load_roll_variables('deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf')

    pitch = head_pose_estimator.return_pitch(img)  # 俯仰角
    yaw = head_pose_estimator.return_yaw(img)  # 偏航角
    roll = head_pose_estimator.return_roll(img)  # 翻转角

    # 输出结果
    print('Estimated pitch...' + str(pitch[0, 0, 0]) + '度')
    print('Estimated yaw...' + str(yaw[0, 0, 0]) + '度')
    print('Estimated roll...' + str(roll[0, 0, 0]) + '度')

    # 返回字典
    return {'pitch_angle': pitch, 'yaw_angle': yaw, 'roll_angle': roll}


if __name__ == '__main__':
    image = cv2.imread('image/waitou.jpg')
    image = cv2.resize(image, dsize=(64, 64))
    get_pitch_yaw_roll(image)
