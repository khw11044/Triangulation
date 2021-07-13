import numpy as np

def rotation_xz(a_,b_,c_):
    a = np.array([a_[9], b_[9], c_[9]])
    len_xy = (a[0] ** 2 + a[1] ** 2) ** 0.5
    ro_z = [[a[1] / len_xy, -a[0] / len_xy, 0],
            [a[0] / len_xy, a[1] / len_xy, 0],
            [0, 0, 1]]
    a_0 = np.dot(ro_z, a)
    len_yz = (a_0[1] ** 2 + a_0[2] ** 2) ** 0.5
    ro_x = [[1, 0, 0],
            [0, a_0[1] / len_yz, a_0[2] / len_yz],
            [0, -a_0[2] / len_yz, a_0[1] / len_yz]]
    for i in range(len(a_)):

        point = np.array([a_[i], b_[i], c_[i]])
        point = np.dot(ro_z, point)
        point = np.dot(ro_x, point)

        a_[i] = point[0]
        b_[i] = point[1]
        c_[i] = point[2]
    return a_, b_, c_

def rotation_y(a_,b_,c_):
    a = np.array([a_[4], b_[4], c_[4]])
    len_xz = (a[0] ** 2 + a[2] ** 2) ** 0.5
    ro_y = [[-a[2] / len_xz, 0, a[0] / len_xz],
            [0, 1, 0],
            [a[0] / len_xz, 0, -a[2] / len_xz]]
    for i in range(len(a_)):
        point = np.array([a_[i], b_[i], c_[i]])
        point = np.dot(ro_y, point)


        a_[i] = point[0]
        b_[i] = point[1]
        c_[i] = point[2]

    return a_, b_, c_