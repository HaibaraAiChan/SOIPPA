

import os

atp = './data/positive'
control = './data/negative'
voxel_folder = './voxel_output/'

list_atp = []
list_control = []
voxel_name_list = []

for filename in os.listdir(voxel_folder):
    if filename:
        voxel_name_list.append(filename[0:-4])

with open(atp) as ad_in:
    for line in ad_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        ttmp = temp.split('\t')
        tmp1 = ttmp[0].split('.')

        aa = tmp1[0]

        res1 = any(aa in voxel for voxel in voxel_name_list)

        if res1:
            list_atp.append(aa)
        else:
            print aa
    list_atp.sort()
    list_atp = list(set(list_atp))
    print list_atp
    print len(list_atp)

ad_in.close()

with open(control) as ot_in:
    for line in ot_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        ttmp = temp.split('\t')

        tmp1 = ttmp[0].split('.')

        aa = tmp1[0]

        res1 = any(aa in voxel for voxel in voxel_name_list)

        if res1:
            list_control.append(aa)
        else:
            print aa

    list_control.sort()
    list_control = list(set(list_control))
    print len(list_control)
ot_in.close()

if os.path.exists("atp"):
    os.remove("atp")
with open("atp", "w") as outf:
    for i in range(len(list_atp)):
        outf.write('%s\n' % list_atp[i])
outf.close()

if os.path.exists("control"):
    os.remove("control")
with open("control", "w") as outf:
    for i in range(len(list_control)):
        outf.write('%s\n' % list_control[i])
outf.close()
