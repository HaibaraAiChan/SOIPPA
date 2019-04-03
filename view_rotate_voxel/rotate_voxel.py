import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from voxel_data_generator import VoxelDataGenerator
import os
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


def plot_orignal_voxel(data):
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

        plt.title('original_' + str(num))
        plt.savefig('./tmp/original_' + str(num))
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
        plt.title('rotate_angle_' + str(axis) + str(rotate_angle) + '_' + str(num))
        plt.savefig('./tmp/rotate_angle_' + str(axis) + str(rotate_angle) + '_' + str(num))
        # plt.show()
        plt.close(fig)
        num = num + 1
        if num == 4:
            break


def save_pickle(out, file, data, rotate_axis, rotate_angle):
    data = next(data)
    data[abs(data) <= 1e-10] = 0.0
    res = data.reshape(1, 14, 32, 32, 32)
    axis = ''
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
    filename = out + file[0:-4] + axis + angle + '.pkl'
    filehandler = open(filename, "wb")
    pickle.dump(res, filehandler)
    filehandler.close()


if __name__ == '__main__':
    input_folder = '../voxel_output/'
    output_folder = '../data_prepare/'

    data_type = 'valid' #'train'  # or 'test', 'valid'
    rotate_axis = 'z'  # ,'y','z'
    rotate_angle = '180'  # ,'090','180','270'




    r_axis = 0
    r_angle = 0

    if rotate_axis == 'x':
        r_axis = 1
    elif rotate_axis == 'y':
        r_axis = 2
    elif rotate_axis == 'z':
        r_axis = 3

    if rotate_angle == '000':
        r_angle = 0
    elif rotate_angle == '090':
        r_angle = 90
    elif rotate_angle == '180':
        r_angle = 180
    elif rotate_angle == '270':
        r_angle = 270

    out = output_folder + data_type + '/' + 'rotate_voxel_data_' + rotate_axis + '_' + rotate_angle+'/'
    # out_ = '../data_prepare/train/rotate_voxel_data_x_000/'
    if not os.path.exists(out):
        os.makedirs(out)

    # out = '../data_prepare/train/rotate_voxel_data_x_090/'

    nummm = 0
    for file in os.listdir(input_folder):
        file_h = open(input_folder + file, 'rb')
        all_data = pickle.load(file_h)
        voxel = all_data[0]
        voxel[:] = voxel[:] - 1
        label = '0'
        # plot_original_voxel(voxel)

        c1 = VoxelDataGenerator(rotate_axis=r_axis, rotate_angle=r_angle)
        data = c1.build(data=voxel, batch_size=14)

        save_pickle(out, file, data, rotate_axis=r_axis, rotate_angle=r_angle)
        # plot_voxel(data, rotate_axis=r_axis, rotate_angle=r_angle)
        # plot_voxel(data, rotate_axis=1, rotate_angle=90)
        # c1 = VoxelDataGenerator(rotate_axis=1, rotate_angle=0)
        # data = c1.build(data=voxel, batch_size=14)
        #
        # save_pickle(out, file, data, rotate_axis=1, rotate_angle=0)
        # plot_voxel(data, rotate_axis=1, rotate_angle=0)
        #

        # c1 = VoxelDataGenerator(rotate_axis=1, rotate_angle=90)
        # data = c1.build(data=voxel, batch_size=14)
        #
        # save_pickle(out, file, data, rotate_axis=1, rotate_angle=90)
        # plot_voxel(data, rotate_axis=1, rotate_angle=90)
        #
        # c1 = VoxelDataGenerator(rotate_axis=1, rotate_angle=180)
        # data = c1.build(data=voxel, batch_size=14)
        #
        # save_pickle(out, file, data, rotate_axis=1, rotate_angle=180)
        # plot_voxel(data, rotate_axis=1, rotate_angle=180)

        # c1 = VoxelDataGenerator(rotate_axis=1, rotate_angle=270)
        # data = c1.build(data=voxel, batch_size=14)
        #
        # save_pickle(out, file, data, rotate_axis=1, rotate_angle=270)
        # plot_voxel(data, rotate_axis=1, rotate_angle=180)
        # c1 = VoxelDataGenerator(rotate_axis=2, rotate_angle=90)
        # data = c1.build(data=voxel, batch_size=14)
        #
        # save_pickle(out, file, data, rotate_axis=2, rotate_angle=90)
        # # plot_voxel(data, rotate_axis=2, rotate_angle=90)

        # c1 = VoxelDataGenerator(rotate_axis=2, rotate_angle=180)
        # data = c1.build(data=voxel, batch_size=14)
        #
        # save_pickle(out, file, data, rotate_axis=2, rotate_angle=180)
        # plot_voxel(data, rotate_axis=1, rotate_angle=90)
        #
        #
        # nummm = nummm+1
        # if nummm>=1:
        #     break

    print 'end'
