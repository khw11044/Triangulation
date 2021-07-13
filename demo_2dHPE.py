import numpy as np

# from get_2d_pose import get_2d_point
from lib.BlazePose import poseDetector
from lib.three_D_transform import estimate_relative_pose_from_correspondence, linear_eigen_triangulation
from lib.rotate import rotate_x,rotate_y,rotate_z
from lib.vis import show3Dpose

from scipy import io
import cv2
import matplotlib.pyplot as plt 
from read_calibration import intrinsic_parameter

# 2D 좌표 함수 
def get_2D(mat,view,frame):
    # 2D 좌표 가져오기 
    full_2Dx = mat['annot2'][view][0][frame][0::2]         # C0 카메라 # 3번째 frame, 0번째부터 끝까지 2씩 증가하면서 가저옴
    full_2Dy = mat['annot2'][view][0][frame][1::2]
    pts = np.array([[mat['annot2'][view][0][frame][0], mat['annot2'][view][0][frame][1]]])  # x,y

    return full_2Dx, full_2Dy, pts



# 3D 좌표 함수
def get_3D(mat,intri_1,intri_2,view_1,view_2,frame):
    full_x_1, full_y_1, pts_1 = get_2D(mat,view_1,frame)
    full_x_2, full_y_2, pts_2 = get_2D(mat,view_2,frame)

    for i in range(28): # 28개 joints
        pts_1 = np.concatenate((pts_1 , np.array([[mat['annot2'][view_1][0][frame][0+2*i], mat['annot2'][view_1][0][frame][1+2*i]]])),axis=0)
        pts_2 = np.concatenate((pts_2 , np.array([[mat['annot2'][view_2][0][frame][0+2*i], mat['annot2'][view_2][0][frame][1+2*i]]])),axis=0)

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

def get2D(image_path):
    image = cv2.imread(image_path)
    detector = poseDetector()

    img = detector.findPose(image)
    lmList, img = detector.findPosition(img)
    x_array = [lmList[0][0]]
    y_array = [lmList[0][1]]
    pts_array = np.array([[lmList[0][0],lmList[0][1]]])
    lmList = lmList[1:]
    for coor in lmList:
        x_array.append(coor[0])
        y_array.append(coor[1])
        # pts_array = np.concatenate((pts_array, np.array([[coor[0],coor[1]]])),axis=0)
    
    full_x = np.array(x_array)
    full_y = np.array(y_array)
    # pts = np.array(pts_array)
    return full_x,full_y, pts_array



# -----------------------------------------------------------------------------------------------------------------------

root = '/media/khw11044/Samsung_T5/Humandataset/mpi_inf_3dhp/S1/Seq2/'
mat = io.loadmat(root + 'annot.mat')

def drawplt(mat,intri_1,intri_2,view_1,view_2,f_start,f_end):
    plt.ion()
    fig = plt.figure(1,figsize=(7,7))
    frame_count = 0
    for frame in range(f_start,f_end):

        file_name_0 = root + 'b_data/video_{0}/video_{1}'.format(view_1,view_1) + '_{0:04d}.png'.format(frame)

        file_name_2 = root + 'b_data/video_{0}/video_{1}'.format(view_2,view_2) + '_{0:04d}.png'.format(frame)

        hpe_x_1,hpe_y_1, pts_array_1 = get2D(file_name_0)   # 2D hpe 
        hpe_x_2,hpe_y_2, pts_array_2 = get2D(file_name_2)

        full_x_1, full_y_1, pts_1 = get_2D(mat,view_1,frame)    # 2D annotation
        full_x_2, full_y_2, pts_2 = get_2D(mat,view_2,frame)


        for i in range(1,20): # 28개 joints
            pts_array_1 = np.concatenate((pts_array_1 , np.array([[hpe_x_1[i], hpe_y_1[i]]])),axis=0)
            pts_array_2 = np.concatenate((pts_array_2 , np.array([[hpe_x_2[i], hpe_y_2[i]]])),axis=0)


        _, R_est, T_est = estimate_relative_pose_from_correspondence(pts_array_2,pts_array_1,intri_2,intri_1)   # three_D_transform.py , R이랑 T를 추정 
        p_nom = np.dot(intri_2,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        proj = np.dot(intri_1,np.concatenate((R_est,T_est),axis=1))  
        two_0 = np.array([hpe_x_1,hpe_y_1],dtype=np.float64)
        two_2 = np.array([hpe_x_2,hpe_y_2],dtype=np.float64)
        new_3d, _ = linear_eigen_triangulation(two_2.T, p_nom, two_0.T, proj)       # three_D_transform.py
        new_3d *=222

        d_3 = new_3d
        d_3 = rotate_x(new_3d,-105)
        # d_3 = rotate_y(d_3,90)
        d_3 = rotate_z(d_3,-90)
        d_3[:,2] += 50 


        fig = plt.figure(1,figsize=(12,12))
        ax = fig.add_subplot('221')
        image = cv2.imread(file_name_0, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.scatter(full_x_1,full_y_1, s=4, c='blue')
        ax.scatter(hpe_x_1,hpe_y_1, s=4, c='red')
        ax.imshow(image)

        ax = fig.add_subplot('222')
        image2 = cv2.imread(file_name_2, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        ax.scatter(full_x_2,full_y_2, s=4, c='blue')
        ax.scatter(hpe_x_2,hpe_y_2, s=4, c='red')
        ax.imshow(image2)

        # 2D HPE
        ax = fig.add_subplot('223', projection='3d', aspect='auto')
        d_3 = rotate_z(d_3,120)
        show3Dpose(d_3, ax, radius=128, mpii=2, lcolor='red')

        # 3D annotation
        ax = fig.add_subplot('224', projection='3d', aspect='auto')
        full_x = mat['annot3'][view_1][0][frame][0::3]
        full_z = -mat['annot3'][view_1][0][frame][1::3]
        full_y = mat['annot3'][view_1][0][frame][2::3]
        coordi = np.array([full_x,full_y,full_z])
        coordi /= 10
        coor = rotate_z(coordi.T,100)
        show3Dpose(coor, ax, radius=128, mpii=1, lcolor='blue')
        # plt.show()
        plt.title(frame, loc='right', pad=5)
        plt.draw()
        plt.savefig('plt_triangulation/{0:04d}.png'.format(frame_count))
        plt.pause(0.01)
        frame_count += 1
        fig.clear()


view_1, view_2 = 5,8 
f_start,f_end = 0, 1999
intri_1, intri_2 = intrinsic_parameter(root,view_1,view_2)
drawplt(mat,intri_1,intri_2,view_1,view_2,f_start,f_end)