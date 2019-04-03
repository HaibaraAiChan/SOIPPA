from __future__ import print_function

import sys
import os
import argparse

import numpy as np

from deepdrug3d import DeepDrug3DBuilder

from keras import callbacks
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.models import load_model
from keras import callbacks

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split



from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras
import tensorflow as tf

config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 20})
sess = tf.Session(config=config)

keras.backend.set_session(sess)
seed = 12308
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alist',
                        required=False,
                        default='atp',
                        help='location of list contains names of proteins binds with atp')
    parser.add_argument('--olist',
                        required=False,
                        default='control',
                        help='location of list contains names of proteins binds with control')
    parser.add_argument('--vfolder',
                        required=False,
                        default='./data_prepare/train/',
                        help='folder for the voxel data')
    parser.add_argument('--bs',
                        required=False,
                        default=64,
                        help='batch size')
    parser.add_argument('--lr',
                        required=False,
                        default=0.00001,
                        help='initial learning rate')
    parser.add_argument('--epoch',
                        required=False,
                        default=200,
                        help='number of epochs for training')
    parser.add_argument('--output',
                        required=False,
                        default=None,
                        help='location for the model to be saved')
    args = parser.parse_args()
    return args


def load_data(atps, controls, voxel_folder, a_times, o_times):
    atp_len = len(atps)
    control_len = len(controls)

    L = atp_len * a_times + control_len * o_times

    voxel = np.zeros(shape=(L, 14, 32, 32, 32),
                     dtype=np.float64)
    label = np.zeros(shape=(L,), dtype=int)
    cnt = 0
    numm = 0

    folder_list = os.listdir(voxel_folder)
    folder_list = [f for f in folder_list if 'rotate_voxel_data' in f]
    folder_list.sort()
    a_f_list = folder_list[0:a_times]
    o_f_list = folder_list[0:o_times]
    print('...Loading the data')
    print(len(a_f_list))
    print(len(o_f_list))

    for folder in a_f_list:
        for filename in os.listdir(voxel_folder + folder):
            ll = filename[0:-4].split('_')
            protein_name = ll[0] + "_" + ll[1]
            full_path = voxel_folder + folder + '/' + filename

            if protein_name in atps:
                temp = np.load(full_path)
                voxel[cnt, :] = temp
                label[cnt] = 1
                cnt = cnt + 1
                numm = numm + 1
                print(numm, end=' ')
                if numm % 20 == 0:
                    print()

    print('the atp list done')
    num = 0
    for folder in o_f_list:
        for filename in os.listdir(voxel_folder + folder):
            ll = filename[0:-4].split('_')
            protein_name = ll[0] + "_" + ll[1]
            full_path = voxel_folder + folder + '/' + filename

            if protein_name in controls:
                temp = np.load(full_path)
                voxel[cnt, :] = temp
                label[cnt] = 0
                cnt = cnt + 1
                num = num + 1
                print(num, end=' ')
                if num % 20 == 0:
                    print()

    print('the control list done')
    print("total " + str(numm + num) + ' ligands')
    return voxel, label


def load_valid_data(atps, controls, voxel_folder, a_times, o_times):
    atp_len = len(atps)
    control_len = len(controls)

    L = atp_len * a_times + control_len * o_times

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

        if protein_name in atps:
            temp = np.load(full_path)
            voxel[cnt, :] = temp
            label[cnt] = 1
            cnt = cnt + 1
            numm = numm + 1
            print(numm, end=' ')
    print('the atp list done')
    num = 0

    for filename in os.listdir(voxel_folder):
        if o_times == 0:
            break
        ll = filename[0:-4].split('_')
        protein_name = ll[0] + "_" + ll[1]
        full_path = voxel_folder + '/' + filename

        if protein_name in controls:
            temp = np.load(full_path)
            voxel[cnt, :] = temp
            label[cnt] = 0
            cnt = cnt + 1
            num = num + 1
            print(num, end=' ')

    print('the control list done')

    print("valid data total " + str(numm + num) + ' ligands')
    return voxel, label


def train_deepdrug(atp_list, control_list, voxel_folder, batch_size, lr, epoch, output):
    voxel_output = './data_prepare/valid/rotate_voxel_data_y_090/'

    mdl = DeepDrug3DBuilder.build()
    # mdl=keras.models.load_model('./deepdrug3d.h5')
    # mdl = multi_gpu_model(mdl,gpus=2)
    print(mdl.summary())

    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

    # We add metrics to get more results you want to see
    mdl.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    # load the data
    atps = []
    with open(atp_list) as atp_in:
        for line in atp_in.readlines():
            temp = line.replace(' ', '').replace('\n', '')
            atps.append(temp)
    controls = []
    with open(control_list) as control_in:
        for line in control_in.readlines():
            temp = line.replace(' ', '').replace('\n', '')
            controls.append(temp)
    # convert data into a single matrix
    a_times = 3  # 3  # the times rotate data needed, rotate 0, rotate 90, rotate 180
    o_times = 2  # 2  # the times rotate data needed, rotate 0, rotate 90
    voxel, label = load_data(atps, controls, voxel_folder, a_times, o_times)
    y = np_utils.to_categorical(label, num_classes=2)

    valid_voxel, valid_label = load_valid_data(atps, controls, voxel_output, 1, 1)
    v_y = np_utils.to_categorical(valid_label, num_classes=2)

    # print(voxel.shape)
    # print(label.shape)
    # print(label)
    earlyStopping = EarlyStopping(monitor='val_loss',
                                  patience=80,
                                  verbose=0,
                                  mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5',
                               save_best_only=True,
                               monitor='val_loss',
                               mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2,
                                       patience=15,
                                       verbose=1,
                                       min_delta=1e-4,
                                       mode='min')



    epo=20
    times = epoch/epo
    print('times:'+str(times))
    for i in range(times):
        print('stage: '+str(i)+'***********************************************')
        id= int(i)
        X_train, X_test, y_train, y_test = train_test_split(voxel, y, test_size=0.2, random_state=id)
        mdl.fit(X_train,
                y_train,
                epochs=epo,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(valid_voxel, v_y),
                #validation_data=(X_test, y_test),
                callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                verbose=2)

        scores = mdl.evaluate(X_test, y_test, verbose=1)
        #scores = mdl.evaluate(valid_voxel, v_y, verbose=1)
        print(scores)


    if output == None:
        mdl.save('deepdrug3d.h5')
    else:
        mdl.save(output)


if __name__ == "__main__":
    args = myargs()
    # args = argdet()
    train_deepdrug(args.alist, args.olist, args.vfolder, args.bs, args.lr, args.epoch, args.output)
