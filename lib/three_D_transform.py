##
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from scipy import io
from rotation import rotation_xz, rotation_y


# ## get key point
# root = '/media/khw11044/Samsung_T5/Humandataset/mpi_inf_3dhp/S1/Seq2/'
# mat = io.loadmat(root + 'annot.mat')



##

def world_to_camera_frame(P,R,T):

    X_cam = R.dot(P - T)

    return X_cam.T

# x_0 = mat['univ_annot3'][0][0][0][0]
# y_0 = mat['univ_annot3'][0][0][0][1]
# z_0 = mat['univ_annot3'][0][0][0][2]
# u_0 = np.array([[x_0],[y_0],[z_0]])

# x_2 = mat['univ_annot3'][2][0][0][0]
# y_2 = mat['univ_annot3'][2][0][0][1]
# z_2 = mat['univ_annot3'][2][0][0][2]
# u_2 = np.array([[x_2],[y_2],[z_2]])

# R_0 = np.array([[0.9650164, 0.00488022, 0.262144],
#        [-0.004488356, -0.9993728, 0.0351275],
#        [0.262151, -0.03507521, -0.9643893]])

# T_0 = np.array([[-562.8666],
#        [1398.138],
#        [3852.623]])

# R_2 = np.array([[ -0.3608179, -0.009492658, 0.932588],
#        [-0.0585942, -0.9977421, -0.03282591],
#        [0.9307939, -0.06648842, 0.359447]])

# T_2 = np.array([[57.25702],
#        [1307.287],
#        [2799.822]])

# dd = world_to_camera_frame(u_0, R_0, T_0)
# ee = world_to_camera_frame(u_2, R_2, T_2)

## 3d 변환
def linear_eigen_triangulation(u1, P1, u2, P2, max_coordinate_value=1.e16):  # two_2.T, p_nom, two_0.T, proj
    x = cv2.triangulatePoints(P1[0:3, 0:4], P2[0:3, 0:4], u1.T, u2.T)  # OpenCV's Linear-Eigen triangl

    x[0:3, :] /= x[3:4, :]  # normalize coordinates
    x_status = (np.max(abs(x[0:3, :]), axis=0) <= max_coordinate_value)  # NaN or Inf will receive status False

    x = x[0:3, :].T
    return x, x_status

# # ex_mat_0 = np.concatenate((R_0,T_0),axis=1)
# #
# # ex_mat_2 = np.concatenate((R_2,T_2),axis=1)
# # pro_0 = np.dot(intri_0,ex_mat_0)
# # pro_2 = np.dot(intri_2,ex_mat_2)
# x2_2 = np.array(mat['annot2'][2][0][0][0])
# y2_2 = np.array(mat['annot2'][2][0][0][1])
# u2_2 = np.array([x2_2, y2_2])
# x2_0 = np.array(mat['annot2'][0][0][0][0])
# y2_0 = np.array(mat['annot2'][0][0][0][1])
# u2_0 = np.array([x2_0, y2_0])
# #
# # sss, _ =linear_eigen_triangulation(u2_2,pro_2,u2_0,pro_0)
# # print(sss)


def estimate_relative_pose_from_correspondence(pts1, pts2, K1, K2): # K : 내부 파라미터 
    f_avg = (K1[0, 0] + K2[0, 0]) / 2
    pts1, pts2 = np.ascontiguousarray(pts1, np.float32), np.ascontiguousarray(pts2, np.float32) # 메모리에 연속 배열 (ndim> = 1)을 반환

    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)    # 왜곡을 없애는 함수
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

    E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),                      # https://www.programcreek.com/python/example/110761/cv2.findEssentialMat
                                   method=cv2.RANSAC, prob=0.999, threshold=3.0 / f_avg)    # E : EssentialMatrix

    points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
    return mask[:, 0].astype(bool), R_est, t_est


# pts_2 = np.array([[mat['annot2'][2][0][0][0], mat['annot2'][2][0][0][1]]])
# pts_0 = np.array([[mat['annot2'][0][0][0][0], mat['annot2'][0][0][0][1]]])
# for i in range(6):
#     pts_2 = np.concatenate((pts_2 , np.array([[mat['annot2'][2][0][0][0+2*i], mat['annot2'][2][0][0][1+2*i]]])),axis=0)
#     pts_0 = np.concatenate((pts_0 , np.array([[mat['annot2'][0][0][0][0+2*i], mat['annot2'][0][0][0][1+2*i]]])),axis=0)
# intri_2 = np.array([[1495.587, 0, 983.8873],
#                     [0, 1497.828, 987.5902],
#                     [0, 0, 1]])
# intri_0 = np.array([[1497.693, 0, 1024.704],
#                     [0, 1497.103, 1051.394],
#                     [0, 0, 1]])
# _, R_est, T_est = estimate_relative_pose_from_correspondence(pts_2,pts_0,intri_2,intri_0)



# p_nom = np.dot(intri_2,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
# proj = np.dot(intri_0,np.concatenate((R_est,T_est),axis=1))

# new_3d_0, _ = linear_eigen_triangulation(u2_2, p_nom, u2_0, proj)
# new_3d_1, _ = linear_eigen_triangulation(u2_2, proj,u2_0, p_nom)


def rotation_xz(a_,b_,c_):
    a = np.array([a_[9], b_[9], c_[9]])
    for i in range(len(a_)):
        point = np.array([a_[i], b_[i], c_[i]])
        len_xy = (a[0] ** 2 + a[1] ** 2) ** 0.5
        ro_z = np.array([[a[1] / len_xy, -a[0] / len_xy, 0],
                [a[0] / len_xy, a[1] / len_xy, 0],
                [0, 0, 1]])
        T = np.array([[-a_[14]],
                      [-b_[14]],
                      -c_[14]])
        world_to_camera_frame(point,ro_z,T)