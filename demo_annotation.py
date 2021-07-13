import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 

from scipy import io
import plotly.graph_objects as go
from lib.three_D_transform import estimate_relative_pose_from_correspondence, linear_eigen_triangulation
from lib.get_3d_skelton import get_3d_skeleton # , show3Dpose
from lib.vis import show3Dpose
from read_calibration import intrinsic_parameter


## get key point
root = '/media/khw11044/Samsung_T5/Humandataset/mpi_inf_3dhp/S1/Seq2/'
mat = io.loadmat(root + 'annot.mat')
frame = 0
view_1 = 7
view_2 = 8

# 내부 파라메타

intri_1, intri_2 = intrinsic_parameter(root,view_1,view_2)

# intri_1 = np.array([[1497.693, 0, 1024.704],
#                     [0, 1497.103, 1051.394],
#                     [0, 0, 1]])

# intri_2 = np.array([[1495.587, 0, 983.8873],
#                     [0, 1497.828, 987.5902],
#                     [0, 0, 1]])


# x축 기준 회전
def rotate_x(d_3,degree):
    for i in range(len(d_3)):
        x = d_3[i][0]
        y = d_3[i][1]
        z = d_3[i][2]

        x_hat = x
        y_hat = y * np.cos(degree*np.pi/180) - z * np.sin(degree*np.pi/180)
        z_hat = y * np.sin(degree*np.pi/180) + z * np.cos(degree*np.pi/180)

        d_3[i][0] = x_hat
        d_3[i][1] = y_hat
        d_3[i][2] = z_hat
    
    return d_3

def rotate_y(d_3,degree):
    for i in range(len(d_3)):
        x = d_3[i][0]
        y = d_3[i][1]
        z = d_3[i][2]

        x_hat = z * np.sin(degree*np.pi/180) + x * np.cos(degree*np.pi/180)
        y_hat = y
        z_hat = z * np.cos(degree*np.pi/180) + x * np.sin(degree*np.pi/180)

        d_3[i][0] = x_hat
        d_3[i][1] = y_hat
        d_3[i][2] = z_hat
    
    return d_3

def rotate_z(d_3,degree):
    for i in range(len(d_3)):
        x = d_3[i][0]
        y = d_3[i][1]
        z = d_3[i][2]

        x_hat = x * np.cos(degree*np.pi/180) - y * np.sin(degree*np.pi/180)
        y_hat = x * np.sin(degree*np.pi/180) + y * np.cos(degree*np.pi/180)
        z_hat = z

        d_3[i][0] = x_hat
        d_3[i][1] = y_hat
        d_3[i][2] = z_hat
    
    return d_3


# 2D 좌표 함수 
def get_2D(mat,view,frame):
    # 2D 좌표 가져오기 
    full_2Dx = mat['annot2'][view][0][frame][0::2]         # C0 카메라 # 3번째 frame
    full_2Dy = mat['annot2'][view][0][frame][1::2]
    pts = np.array([[mat['annot2'][view][0][frame][0], mat['annot2'][view][0][frame][1]]])  

    return full_2Dx, full_2Dy, pts



# 3D 좌표 함수
def get_3D(mat,intri_1,intri_2,view_1,view_2,frame):
    full_x_1, full_y_1, pts_1 = get_2D(mat,view_1,frame)
    full_x_2, full_y_2, pts_2 = get_2D(mat,view_2,frame)

    for i in range(28): # 28개 joints
        pts_1 = np.concatenate((pts_1 , np.array([[full_x_1[i], full_y_1[i]]])),axis=0)
        pts_2 = np.concatenate((pts_2 , np.array([[full_x_2[i], full_y_2[i]]])),axis=0)
        
    _, R_est, T_est = estimate_relative_pose_from_correspondence(pts_2,pts_1,intri_2,intri_1)   # three_D_transform.py , R이랑 T를 추정 
    p_nom = np.dot(intri_2,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    proj = np.dot(intri_1,np.concatenate((R_est,T_est),axis=1))  
    two_0 = np.array([full_x_1,full_y_1])
    two_2 = np.array([full_x_2,full_y_2])
    new_3d, _ = linear_eigen_triangulation(two_2.T, p_nom, two_0.T, proj)       # three_D_transform.py
    new_3d *=350 
    
    d_3 = new_3d
    d_3 = rotate_x(new_3d,-105)
    # d_3 = rotate_y(d_3,90)
    d_3 = rotate_z(d_3,-90)

    return [full_x_1,full_y_1], [full_x_2,full_y_2], d_3







# # 보여주기
def drawplt(mat,intri_1,intri_2,view_1,view_2,f_start,f_end):
    plt.ion()
    fig = plt.figure(1,figsize=(7,7))
    frame_count = 0
    for frame in range(f_start,f_end):
        view1_xy, view2_xy, new_3d = get_3D(mat,intri_1,intri_2,view_1,view_2,frame)

        
        ax = fig.add_subplot('221')
        image_path = root + 'b_data/video_{0}/video_{1}'.format(view_1,view_1) + '_{0:04d}.png'.format(frame)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.scatter(view1_xy[0],view1_xy[1], s=4, c='red')
        ax.imshow(image)

        
        ax = fig.add_subplot('222')
        image_path2 = root + 'b_data/video_{0}/video_{1}'.format(view_2,view_2) + '_{0:04d}.png'.format(frame)
        image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        ax.scatter(view2_xy[0],view2_xy[1], s=4, c='red')
        ax.imshow(image2)

        ax = fig.add_subplot('223', projection='3d', aspect='auto')
        show3Dpose(new_3d, ax, radius=128,lcolor='red')
        # ax.view_init(120, 90)
        
        ax = fig.add_subplot('224', projection='3d', aspect='auto')
        full_x = mat['annot3'][view_1][0][frame][0::3]
        full_z = -mat['annot3'][view_1][0][frame][1::3]
        full_y = mat['annot3'][view_1][0][frame][2::3]
        coordi = np.array([full_x,full_y,full_z])
        coordi /= 10
        show3Dpose(coordi.T, ax, radius=128, lcolor='blue')
        # plt.show()
        plt.title(frame, loc='right', pad=5)
        plt.draw()
        plt.pause(0.01)
        # fig.tight_layout()
        plt.savefig('plt_triangulation/{0:04d}.png'.format(frame_count))
        frame_count += 1
        fig.clear()

f_start,f_end = 0, 1999
drawplt(mat,intri_1,intri_2,view_1,view_2,f_start,f_end)

# # 2048, 2048
# fig = plt.figure(figsize=(12,7))

# ax = fig.add_subplot('221')
# img_path = root + 'b_data/video_7/video_7_0000.png'
# image = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# ax.scatter(full_x_0,full_y_0, s=4, c='red')
# ax.imshow(image)

# ax = fig.add_subplot('222')
# img_path2 = root + 'b_data/video_8/video_8_0000.png'
# image2 = cv2.imread(img_path2, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# ax.scatter(full_x_2,full_y_2, s=4, c='red')
# ax.imshow(image2)

# ax = fig.add_subplot('223', projection='3d', aspect='auto')
 
# show3Dpose(new_3d, ax, radius=128)
# ax.view_init(-75, -90)

# ax = fig.add_subplot('224', projection='3d', aspect='auto')
# full_x = mat['annot3'][view_1][0][check_frame][0::3]
# full_z = -mat['annot3'][view_1][0][check_frame][1::3]
# full_y = mat['annot3'][view_1][0][check_frame][2::3]
   
# show3Dpose(new_3d, ax, radius=128)
# ax.view_init(-75, -90)

# plt.show()

# 로컬 웹에서 보기 
# get_3d_skeleton(new_3d[:,0], new_3d[:,1], new_3d[:,2])                      # get_3d_skelton.py