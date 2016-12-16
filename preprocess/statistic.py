#!/usr/bin/env python3

def label_statistic(path):
    print("Reading path " + path + " ...")
    word_dic = {}
    with open(path) as f:
        for line in f:
            label, _ = line.strip().split("\t")
            if label in word_dic.keys():
                word_dic[label] += 1
            else:
                word_dic[label] = 1
    for k,v in word_dic.items():
        print("\t".join([k, str(v)]))
    print("")
