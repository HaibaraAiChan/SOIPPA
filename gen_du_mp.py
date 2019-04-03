import sys
import os
import argparse
import numpy as np
import shutil
import cPickle
from voxelization import Vox3DBuilder
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")


def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein',
                        required=False,
                        default='./data/Proteins/',
                        help='location of the protein pdb file path')
    parser.add_argument('--aux',
                        required=False,
                        default='./data/aux_files/',
                        help='location of the auxilary input file')
    parser.add_argument('--r',
                        required=False,
                        help='radius of the grid to be generated',
                        default=15,
                        type=int,
                        dest='r')
    parser.add_argument('--N',
                        required=False,
                        help='number of points long the dimension the generated grid',
                        default=31,
                        type=int,
                        dest='N')
    parser.add_argument('--output',
                        required=False,
                        default='./voxel_output/',
                        help='location of the protein pdb file path')
    args = parser.parse_args()
    return args


def gen_one_voxel(protein_path, aux_path, output, r, N):
    voxel = Vox3DBuilder.voxelization(protein_path, aux_path, r, N)
    print "the type of voxel data--------------------------------"
    print type(voxel)
    print voxel.shape
    print "the type of voxel data--------------------------------"
    if not os.path.exists(output):
        os.makedirs(output)
    outname = os.path.splitext(os.path.basename(aux_path))[0][0:-4]

    oname = output + outname + ".pkl"
    # if os.path.exists(oname):
    #    os.remove(oname)
    cPickle.dump(voxel, open(oname, "wb"))
    # Y = cPickle.load(open(oname, "rb"))
    # print Y[0][0][0][0]

    print('The end of one file ')


def worker(aux_filename_list, args):
    protein_path = args.protein
    aux_path = args.aux

    for aux_file in aux_filename_list:
        base = aux_file[0:-4].split('_')
        pro_file = base[0] + '.pdb'
        print "PID:		 " + str(os.getpid())
        print "protein file: " + pro_file
        print "aux_file:     " + aux_file

        gen_one_voxel(protein_path + pro_file, aux_path + aux_file, args.output, args.r, args.N)


if __name__ == "__main__":
    args = myargs()

    protein_path = args.protein
    aux_path = args.aux
    aux_filename_list = []
    protein_list = []
    #    for aux_file in os.listdir(aux_path):
    #        if aux_file:
    #            aux_filename_list.append(aux_file)
    #    print "the number of aux files is " + str(len(aux_filename_list))
    #    aux_filename_list = ['2b0qA_ADP_aux.txt', '1ksfX_ADP_aux.txt', '1yoaA_FMN_aux.txt', '1xcjA_SAH_aux.txt', '1yqtA_ADP_aux.txt', '1xmvA_ADP_aux.txt', '2d2fA_ADP_aux.txt', '1zr6A_FAD_aux.txt', '1v93A_FAD_aux.txt', '1w1oA_FAD_aux.txt', '1xngA_ATP_aux.txt', '2dpmA_SAM_aux.txt', '2aruA_ATP_aux.txt', '1w78A_ADP_aux.txt', '2b9wA_FAD_aux.txt', '1z2nX_ADP_aux.txt', '2c5eA_NAD_aux.txt', '1u3eM_TRS_aux.txt', '1v1aA_ADP_aux.txt', '1wbpA_ADP_aux.txt', '2c2aA_ADP_aux.txt', '1yf3A_SAH_aux.txt', '1ybhA_FAD_aux.txt', '1kxhA_ACR_aux.txt', '1krhA_FAD_aux.txt', '1wmdA_SO4_aux.txt', '1vi2A_NAD_aux.txt', '1u3eM_EDO_aux.txt', '1u1jA_SO4_aux.txt', '2b69A_NAD_aux.txt', '1xtjA_ADP_aux.txt', '1yp4A_ADP_aux.txt', '1u8xX_NAD_aux.txt', '1wm1A_PTB_aux.txt', '1wxhA_NAD_aux.txt', '2a14A_SAH_aux.txt', '1y8oA_ADP_aux.txt', '1y56A_ATP_aux.txt', '2a2cA_ADP_aux.txt', '2bh2A_SAH_aux.txt', '1uj2A_ADP_aux.txt', '1u1jA_MSE_aux.txt', '1u1jA_C2F_aux.txt', '1xdnA_ATP_aux.txt', '1vjpA_NAD_aux.txt', '1ytmA_ATP_aux.txt', '1vm6C_NAD_aux.txt', '1z6lA_FAD_aux.txt', '2bzgA_SAH_aux.txt', '1kqfA_MGD_aux.txt', '1wpeA_ADP_aux.txt', '1vqwA_FAD_aux.txt', '1va6A_ADP_aux.txt', '1vrqA_NAD_aux.txt', '1vbiA_NAD_aux.txt', '1kqfA_6MO_aux.txt', '2hgsA_ADP_aux.txt', '2fpkA_ADP_aux.txt', '2cvjA_FAD_aux.txt', '1xhcA_FAD_aux.txt', '2uagA_ADP_aux.txt', '1ku0A_CA_aux.txt', '1u1jA_MET_aux.txt', '2aotA_SAH_aux.txt', '2bi7A_FAD_aux.txt', '1um8A_ADP_aux.txt', '1kzpB_ACY_aux.txt', '1yoaA_FAD_aux.txt', '1zaoA_ATP_aux.txt', '1y63A_ADP_aux.txt', '1wmdA_GOL_aux.txt', '1wdkA_NAD_aux.txt', '1xw4X_ADP_aux.txt', '1v5eA_FAD_aux.txt', '2b9eA_SAM_aux.txt', '1xzlA_HEE_aux.txt', '1u0xA_HEM_aux.txt', '2bgiA_FAD_aux.txt', '8icnA_ATP_aux.txt', '1vc9A_ATP_aux.txt', '1u0jA_ADP_aux.txt', '1vkoA_NAD_aux.txt', '1kqwA_RTL_aux.txt', '1uakA_SAM_aux.txt', '2aqjA_FAD_aux.txt', '2bfrA_ADP_aux.txt', '1v2xA_SAM_aux.txt', '1kqfA_SF4_aux.txt', '1uwkA_NAD_aux.txt', '1wklA_ADP_aux.txt', '1u3gA_ADP_aux.txt', '1yq0A_ADP_aux.txt', '2cwwA_SAH_aux.txt', '2f8lA_SAM_aux.txt', '1v7rA_CIT_aux.txt', '1ubxA_FPP_aux.txt', '1l3rE_ADP_aux.txt', '1kxgA_CIT_aux.txt', '4at1D_ATP_aux.txt', '1w2dA_ADP_aux.txt', '1u5vA_ATP_aux.txt', '1v97A_FAD_aux.txt', '1wmdA_DIO_aux.txt', '2f02A_ATP_aux.txt', '2fg9A_FAD_aux.txt', '1y56A_FMN_aux.txt', '2a5fB_NAD_aux.txt', '2culA_FAD_aux.txt', '1ymfA_ADP_aux.txt', '1w4xA_FAD_aux.txt', '1kvkA_ATP_aux.txt', '2axnA_ADP_aux.txt']

    aux_filename_list = [

        '1kqfA_6MO_aux.txt',
        '1kqfA_MGD_aux.txt', '1kqfA_SF4_aux.txt']
    print "the number of aux files is " + str(len(aux_filename_list))

    size = len(aux_filename_list)

    P_NUM = 1
    p = Pool(P_NUM)
    for i in range(P_NUM):
        p.apply_async(worker, args=(aux_filename_list[size / P_NUM * i: size / P_NUM * (i + 1)], args,))

    p.close()
    p.join()


