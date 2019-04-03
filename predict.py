
from __future__ import print_function

import sys
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
# from voxelization import Vox3DBuilder
import pickle
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc
from keras.models import load_model
from sklearn import metrics


def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein',
                        required=True,
                        help='location of the protein pdb file path')
    parser.add_argument('--aux',
                        required=True,
                        help='location of the auxilary input file')
    parser.add_argument('--r',
                        required=False,
                        help='radius of the grid to be generated', default=15,
                        type=int,
                        dest='r')
    parser.add_argument('--N',
                        required=False,
                        help='number of points long the dimension the generated grid', default=31,
                        type=int,
                        dest='N')
    args = parser.parse_args()
    return args


def load_valid_data(adenines, others, voxel_folder, a_times, o_times):
    adenine_len = len(adenines)
    other_len = len(others)

    L = adenine_len * a_times + other_len * o_times

    voxel = np.zeros(shape=(L, 14, 32, 32, 32),
                     dtype=np.float64)
    label = np.zeros(shape=(L,), dtype=int)
    cnt = 0
    numm = 0
    print('...Loading valid data')

    for filename in os.listdir(voxel_folder):
        if a_times == 0:
            break
        ll = filename[0:-4].split('_')
        protein_name = ll[0] + "_" + ll[1]
        full_path = voxel_folder + '/' + filename

        if protein_name in adenines:
            temp = np.load(full_path)
            voxel[cnt, :] = temp
            label[cnt] = 1
            cnt = cnt + 1
            numm = numm + 1
            print(numm, end=' ')
            if numm % 20 == 0:
                print()

    print('the adenine list done')
    num = 0

    for filename in os.listdir(voxel_folder):
        if o_times == 0:
            break
        ll = filename[0:-4].split('_')
        protein_name = ll[0] + "_" + ll[1]
        full_path = voxel_folder + '/' + filename

        if protein_name in others:
            temp = np.load(full_path)
            voxel[cnt, :] = temp
            label[cnt] = 0
            cnt = cnt + 1
            num = num + 1
            print(num, end=' ')
            if num % 20 == 0:
                print()

    print('the other list done')

    print("valid data total " + str(numm + num) + ' ligands')
    return voxel, label


def predict(path, adenine_list, other_list):

    # load the data
    adenines = []
    with open(adenine_list) as adenine_in:
        for line in adenine_in.readlines():
            temp = line.replace(' ', '').replace('\n', '')
            adenines.append(temp)
    others = []
    with open(other_list) as other_in:
        for line in other_in.readlines():
            temp = line.replace(' ', '').replace('\n', '')
            others.append(temp)

    valid_voxel, valid_label = load_valid_data(adenines, others, path, 1, 1)
    v_y = np_utils.to_categorical(valid_label, num_classes=2)

    mdl = load_model('./deepdrug3d.h5')

    score = mdl.predict(valid_voxel)

    # Compute ROC curve and ROC area for each class
    n_classes = 2

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        y_score = np.array(score[:, i])
        y_test = np.array(v_y[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score, pos_label=1)
        # fpr_t, tpr_t, _t = metrics.roc_curve(y_test, y_score, pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(v_y.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    # plt.plot(fpr[0], tpr[0], color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SOIPPA')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # args = myargs()
    # main(args.protein, args.aux, args.r, args.N)
    adenine = 'atp'
    other = 'control'
    # path = './data_prepare/valid/rotate_voxel_data_y_090/'#### 0.74
    path = './data_prepare/valid/rotate_voxel_data_y_180/'## 0.80
    # path = './data_prepare/valid/rotate_voxel_data_z_180/'
    predict(path,adenine,other)
