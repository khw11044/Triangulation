import shutil
import numpy as np

# root = '/media/khw11044/Samsung_T5/Humandataset/mpi_inf_3dhp/S1/Seq2'


def intrinsic_parameter(root,view_1,view_2):
    camera_calibration = root + '/camera.calibration'
    shutil.copy(camera_calibration, root + '/camera.txt')
    f = open(root + '/camera.txt', 'r')
    lines = f.readlines()
    view_num = 0
    # i : 1, 8, 15, 22
    for i, line in enumerate(lines):
        if i % 7 == 1:
            if view_num == view_1:
                intri_view_1 = [x for x in lines[i+4].split(' ') if x != '' and x != '\n' and x != 'intrinsic' ]
                intri_view_1 = list(map(float, intri_view_1))
                print(intri_view_1)
            if view_num == view_2:
                intri_view_2 = [x for x in lines[i+4].split(' ') if x != '' and x != '\n' and x != 'intrinsic' ]
                intri_view_2 = list(map(float, intri_view_2))
                print(intri_view_2)
            view_num += 1
    f.close()

    intri_1 = np.array([[intri_view_1[0], intri_view_1[1], intri_view_1[2]],
                    [intri_view_1[4], intri_view_1[5], intri_view_1[6]],
                    [intri_view_1[8], intri_view_1[9], intri_view_1[10]]])

    intri_2 = np.array([[intri_view_2[0], intri_view_2[1], intri_view_2[2]],
                    [intri_view_2[4], intri_view_2[5], intri_view_2[6]],
                    [intri_view_2[8], intri_view_2[9], intri_view_2[10]]])

    return intri_1, intri_2

# intri_1, intri_2 = intrinsic_parameter(root,7,8)

# print(intri_1)
