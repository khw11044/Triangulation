##
import numpy as np
import cv2
import os

from scipy import io
import plotly.graph_objects as go
from rotation import rotation_xz, rotation_y


# 3D 좌표화 
def get_3d_skeleton(full_x,full_y,full_z):
    # full_z = -full_y
    # full_y = full_z
    connect = [[7,6],[6,5],[5,9],[5,14],[9,10],[10,11],[11,12],[14,15],[15,16],[16,17],[5,4],[4,24],[24,25],[25,26],[4,19],[19,20],[20,21]]
    x = []
    y = []
    z = []
    for [i,j] in connect:
        x_c = [full_x[i],full_x[j], None]
        y_c = [full_y[i],full_y[j], None]
        z_c = [full_z[i],full_z[j], None]
        x.extend(x_c)
        y.extend(y_c)
        z.extend(z_c)

    x_range = max(full_x)-min(full_x)
    y_range = max(full_y)-min(full_y)
    z_range = max(full_z)-min(full_z)

    max_range_half = int(max([x_range, y_range, z_range])/2)

    x_center = int((max(full_x)+min(full_x))/2)
    y_center = int((max(full_y)+min(full_y))/2)
    z_center = int((max(full_z)+min(full_z))/2)

    x_range = [x_center-max_range_half-100, x_center+max_range_half+100]
    z_range = [z_center-max_range_half-100, z_center + max_range_half+100]
    y_range = [y_center-max_range_half-100, y_center+max_range_half+100]

    fig= go.Figure(go.Scatter3d(x= x,
                                y= z,
                                z= y,
                                    mode='lines',
                                    line_width=2,
                                    line_color='blue'
                                    ))

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=x_range),
                         zaxis = dict(nticks=4, range=z_range),
                         yaxis = dict(nticks=4, range=y_range),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))

    fig.show()


def show3Dpose(full_x,full_y,full_z, ax, radius=40):
    connect = [[7,6],[6,5],[5,9],[5,14],[9,10],[10,11],[11,12],[14,15],[15,16],[16,17],[5,4],[4,24],[24,25],[25,26],[4,19],[19,20],[20,21]]
    x = []
    y = []
    z = []
    for [i,j] in connect:
        x_c = [full_x[i],full_x[j], None]
        y_c = [full_y[i],full_y[j], None]
        z_c = [full_z[i],full_z[j], None]
        x.extend(x_c)
        y.extend(y_c)
        z.extend(z_c)

    x_range = max(full_x)-min(full_x)
    y_range = max(full_y)-min(full_y)
    z_range = max(full_z)-min(full_z)

    max_range_half = int(max([x_range, y_range, z_range])/2)

    x_center = int((max(full_x)+min(full_x))/2)
    y_center = int((max(full_y)+min(full_y))/2)
    z_center = int((max(full_z)+min(full_z))/2)

    RADIUS = radius  # space around the subject

    x_range = [x_center-max_range_half-RADIUS, x_center+max_range_half+RADIUS]
    z_range = [z_center-max_range_half-RADIUS, z_center + max_range_half+RADIUS]
    y_range = [y_center-max_range_half-RADIUS, y_center+max_range_half+RADIUS]

    
    ax.set_xlim3d(x_range)
    ax.set_zlim3d(z_range)
    ax.set_ylim3d(y_range)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def main():
    ## get key point
    root = 'S1/Seq1/'

    mat = io.loadmat(root + 'annot.mat')


    full_x = mat['annot3'][0][0][0][0::3]
    full_z = -mat['annot3'][0][0][0][1::3]
    full_y = mat['annot3'][0][0][0][2::3]

    # # ..
    # full_x = full_x - full_x[14]
    # full_y = full_y - full_y[14]
    # full_z = full_z - full_z[14]

    # full_x, full_y, full_z = rotation_xz(full_x,full_y,full_z)
    # full_x, full_y, full_z = rotation_y(full_x,full_y,full_z)
    get_3d_skeleton(full_x,full_y,full_z)

if __name__ == "__main__":
	main()