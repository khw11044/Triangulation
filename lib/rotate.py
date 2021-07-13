import numpy as np

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