import itertools
import numpy as np
import os
import sys
pet_labels = [["yes", "no"], ["true", "false"]]
pet_patterns = [
    ["[PARAGRAPH]", " Question : [QUESTION] ? Answer : {}. [SEP]".format('MASK'), ""],
    ["[PARAGRAPH]", " Based on the previous passage, [QUESTION] ? {}. [SEP]".format('MASK'), ""],
    ["Based on the following passage, [QUESTION] ? {}. ".format('MASK'), " [PARAGRAPH] [SEP]", ""]]

pet_pvps = list(itertools.product(pet_patterns, pet_labels))
for per in pet_pvps:
    print(per)

# bs = 1
# r = np.ones((bs, 1)) * 256
# print(r)
path = './exp_out/'
for dir1 in os.listdir(path):
    path1 = os.path.join(path,dir1)
    for di2 in os.listdir(path1):
        path2 = os.path.join(path1,di2)
        for di3 in os.listdir(path2):
            if di3=='best_model.pt' or di3=='final_model.pt':
                print('-----------')
                os.system('sudo rm -rf {}'.format(di3))