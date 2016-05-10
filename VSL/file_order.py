#!/usr/local/bin/python

import numpy as np
import json


def main():
    f = open('MasterFileOrder.txt', mode='r')
    lines = f.readlines()
    f.close()

    sub1 = [l.replace('yaleB01_P00', '').replace('_bgCorrected.tif\n', '') for l in lines if 'yaleB01' in l]
    print sub1

    E = list()
    A = list()

    for i in sub1:
        E.append(i.split('E')[1])
        A.append(i.split('E')[0].replace('A', ''))

    E = map(int, list(set(E)))
    A = map(int, list(set(A)))

    # E.sort(reverse=True)
    # A.sort(reverse=True)
    E.sort(reverse=False)
    A.sort(reverse=False)

    E = np.array(E)
    A = np.array(A)

    print E, 'len:', E.__len__()
    print A, 'len:', A.__len__()

    mapping_dict = dict()
    for i in range(sub1.__len__()):
        e = int(sub1[i].split('E')[1])
        a = int(sub1[i].split('E')[0].replace('A', ''))

        print i, [np.where(E == e)[0][0], np.where(A == a)[0][0]]
        mapping_dict[i] = [np.where(E == e)[0][0], np.where(A == a)[0][0]]

    print mapping_dict

    with open('mapping_dict.json', mode='w') as f2:
        json.dump(mapping_dict, f2)

if __name__ == '__main__':
    main()
