import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from voxel_data_generator import VoxelDataGenerator
import pickle

def floatRgb(mag, cmin, cmax):
    """ Return a tuple of floats between 0 and 1 for R, G, and B. """
    # Normalize to 0-1
    try:
        x = float(mag - cmin) / (cmax - cmin)
    except ZeroDivisionError:
        x = 0.5  # cmax == cmin
    blue = min((max((4 * (0.75 - x), 0.)), 1.))
    red = min((max((4 * (x - 0.25), 0.)), 1.))
    green = min((max((4 * math.fabs(x - 0.5) - 1., 0.)), 1.))
    return red, green, blue


def plot_original_voxel(data):

    num = 1
    # data = next(data)
    data[abs(data) <= 1e-10] = 0.0

    for v in data:
        fig = plt.figure(figsize=plt.figaspect(1.0))
        spatial_axes = [32, 32, 32]
        filled = np.ones(spatial_axes, dtype=np.bool)
        cmax = v.max()
        print cmax
        cmin = v.min()
        print cmin
        print"#######################"
        for i in range(32):
            for j in range(32):
                for k in range(32):

                    if abs(v[i][j][k]) <= 1e-10:
                        filled[i][j][k] = False

            print v[i].max()
            print v[i].min()
        colors = np.empty(spatial_axes + [4], dtype=np.float32)

        alpha = .3

        for i in range(32):
            for j in range(32):
                for k in range(32):
                    r, g, b = floatRgb(v[i][j][k], cmin, cmax)
                    color = [r, g, b, alpha]
                    colors[i][j][k] = color

        # and plot everything

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax = fig.gca(projection='3d')
        ax.voxels(filled,
                  facecolors=colors,
                  edgecolors='k',
                  linewidth=0.1)
        ax.set(xlabel='r', ylabel='g', zlabel='b')

        plt.title('original_'+str(num))
        plt.savefig('./tmp/original_'+str(num))
        # plt.show()
        plt.close(fig)
        num = num + 1
        if num == 4:
            break


def plot_voxel(data, rotate_axis, rotate_angle):

    num = 1
    data = next(data)
    data[abs(data) <= 1e-10] = 0.0

    for v in data:
        fig = plt.figure(figsize=plt.figaspect(1.0))
        spatial_axes = [32, 32, 32]
        filled = np.ones(spatial_axes, dtype=np.bool)
        cmax = v.max()
        print cmax
        cmin = v.min()
        print cmin
        print"#######################"
        for i in range(32):
            for j in range(32):
                for k in range(32):

                    if abs(v[i][j][k]) <= 1e-10:
                        filled[i][j][k] = False

            print v[i].max()
            print v[i].min()
        colors = np.empty(spatial_axes + [4], dtype=np.float32)

        alpha = .3
        idx = 0
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    r, g, b = floatRgb(v[i][j][k], cmin, cmax)
                    color = [r, g, b, alpha]
                    colors[i][j][k] = color

        # and plot everything

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax = fig.gca(projection='3d')
        ax.voxels(filled,
                  facecolors=colors,
                  edgecolors='k',
                  linewidth=0.1)
        ax.set(xlabel='r', ylabel='g', zlabel='b')
        if rotate_axis == 1:
            axis = 'x_'
        elif rotate_axis == 2:
            axis = 'y_'
        elif rotate_axis == 3:
            axis = 'z_'
        plt.title('rotate_angle_'+str(axis)+str(rotate_angle)+'_'+str(num))
        plt.savefig('./tmp/rotate_angle_'+str(axis)+str(rotate_angle)+'_'+str(num))
        # plt.show()
        plt.close(fig)

        num = num + 1
        if num == 4:
            break
    plt.show()


def save_pickle(out, file, data, rotate_axis, rotate_angle):
    data = next(data)
    data[abs(data) <= 1e-10] = 0.0
    res = data.reshape(1, 14, 32, 32, 32)
    axis=''
    if rotate_axis == 1:
        axis = '_x_'
    elif rotate_axis == 2:
        axis = '_y_'
    elif rotate_axis == 3:
        axis = '_z_'
    angle = ''
    if rotate_angle == 0:
        angle = 'r_0'
    elif rotate_angle == 90:
        angle = 'r_90'
    elif rotate_angle == 180:
        angle = 'r_180'
    elif rotate_angle == 270:
        angle = 'r_270'
    filename = out+file[0:-4] + axis+angle + '.pkl'
    filehandler = open(filename, "wb")
    pickle.dump(res, filehandler)
    filehandler.close()



if __name__ == '__main__':
    # all_data = np.load('../voxel_output/1a0iA_ATP.pkl')
    # all_data = np.load('../data_prepare/train/rotate_voxel_data_x_000/1t3tA_ADP_x_r_0.pkl')
    # all_data = np.load('../data_prepare/train/rotate_voxel_data_x_000/1a0iA_ATP_x_r_0.pkl')
    all_data = np.load('../data_prepare/test/rotate_voxel_data_y_000/1a0iA_ATP_y_r_0.pkl')

    out = './rotate_voxel_data/'
    voxel = all_data[0]
    # voxel[:] = voxel[:] - 1
    label = '0'

    plot_original_voxel(voxel)

    # c1 = VoxelDataGenerator(rotate_axis=1, rotate_angle=0)
    # data = c1.build(data=voxel, batch_size=14)
    #
    # # save_pickle(out, file, data, rotate_axis=1, rotate_angle=0)
    # plot_voxel(data, rotate_axis=1, rotate_angle=0)



    # plot_original_voxel(voxel)

    # for vox in voxel:

    # c1 = VoxelDataGenerator(rotate_axis=1, rotate_angle=0)
    # g1 = c1.build(data=voxel, batch_size=14)
    #
    # plot_voxel(g1, rotate_axis=1, rotate_angle=0)

    # c1 = VoxelDataGenerator(rotate_axis=2, rotate_angle=0)
    # g1 = c1.build(data=voxel, batch_size=14)
    #
    # plot_voxel(g1, rotate_axis=2, rotate_angle=0)
    # c2 = VoxelDataGenerator(rotate_axis=1, rotate_angle=90)
    # g2 = c2.build(data=voxel, batch_size=1)
    # plot_voxel(g2, rotate_angle=90)
    #
    # c4 = VoxelDataGenerator(rotate_axis=1, rotate_angle=180)
    # g4 = c4.build(data=voxel, batch_size=1)
    # plot_voxel(g4, rotate_angle=180)
    print 'end'

