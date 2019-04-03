import os
import shutil
import numpy as np
from biopandas.pdb import PandasPdb


def gen_one_aux_file(pocket_path, pocket_name, protein_name, output_path):
    base = pocket_name.split('.')
    pro_base = protein_name.split('.')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ppdb = PandasPdb().read_pdb(pocket_path + pocket_name)
    protein_df = ppdb.df['ATOM']

    protein_res_ID = protein_df['residue_number']
    residue_list, count = np.unique(protein_res_ID, return_counts=True)

    if len(residue_list) <= 2:
        return False

    f = open(output_path + pro_base[0] + '_' + base[1] + '_aux.txt', 'w')
    f.write('BindingResidueIDs:')
    for i in range(len(residue_list)):

        if i < len(residue_list) - 1:
            f.write(str(residue_list[i]) + " ")
        else:
            f.write(str(residue_list[i]) + "\n")
    f.write('BindingSiteCenter:')
    f.close()
    return True

def gen_all_aux_file_247(pocket_path, protein_path, output_path):

    protein_list=[]
    for filename in os.listdir(protein_path):
        if filename:
            protein_list.append(filename)
    num = 0
    for filename in os.listdir(pocket_path):
        sub = filename.split('.')[0]
        # sub = sub[0:-2]
        sub_list = [s for s in protein_list if sub in s]
        if len(sub_list) == 0:
            print filename+'*********************************'
        elif len(sub_list) == 2:
            print sub_list
        else:
            # print sub_list[0]
            num = num+1
            tmp = gen_one_aux_file(pocket_path, filename, sub_list[0], output_path)
            if tmp == False:
                print(filename)
    print num

def gen_all_aux_file_101(pocket_path, protein_path, output_path):
    protein_list=[]
    for filename in os.listdir(protein_path):
        if filename:
            protein_list.append(filename)
    num = 0
    for filename in os.listdir(pocket_path):
        sub = filename.split('.')[0]
        sub = sub[0:-2]
        sub_list = [s for s in protein_list if sub in s]
        if len(sub_list) == 0:
            print filename+'*********************************'
        elif len(sub_list) == 2:
            print sub_list
        else:
            # print sub_list[0]
            num = num+1
            tmp = gen_one_aux_file(pocket_path, filename, sub_list[0], output_path)
            if tmp == False:
                print(filename)
    print num


if __name__ == "__main__":
    pocket_path = '../data/original_data/pockets-101_SP/'
    protein_path = '../data/original_data/protein-101/'
    output_path = '../data/original_data/101/'
    gen_all_aux_file_101(pocket_path=pocket_path, protein_path=protein_path, output_path=output_path)

    # pocket_path = '../data/original_data/pockets-247_SP/'
    # protein_path = '../data/original_data/protein-247/'
    # output_path = '../data/original_data/247/'
    # gen_all_aux_file_247(pocket_path=pocket_path, protein_path=protein_path, output_path=output_path)

    # shutil.rmtree(output_path)
