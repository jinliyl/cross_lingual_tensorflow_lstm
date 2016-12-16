# -*- coding: utf-8 -*-

class make_index:
    
    def __init__(self, data_path, label_path, dic_path):
        self.data_path = data_path
        self.label_path = label_path
        self.dic_path = dic_path
        
        self.word_dic = {}
        self.label_emotion_dic = {}

    def load_word_dic(self):
        with open(self.dic_path) as f:
            for line in f:
                line = line.strip()
                word, count = line.split("\t")
                self.word_dic[word] = count
    
    def load_label_dic(self):
        with open(self.label_path) as f:
            for line in f:
                line = line.strip()
                label, emotion = line.split(" ")
                self.label_emotion_dic[label] = emotion

    def load_save_comment(self):
        with open(self.data_path) as f, open(self.data_path + "_index", "w") as fw:
            for line in f:
                line = line.strip()
                label, content = line.split("\t")
                if label not in self.label_emotion_dic.keys():
                    #print("error in label: " + str(label))
                    continue
                emotion = self.label_emotion_dic[label]
                word_list = []
                for word in content.split(" "):
                    word_list.append(str(self.word_dic[word]))
                print("\t".join([emotion, " ".join(word_list)]), file = fw)

    def process_index(self):
        self.load_word_dic()
        self.load_label_dic()
        self.load_save_comment()

def transform(en_path, cn_path, trans_dic_path, output_path):
        en_dic = {}
        cn_dic = {}

        with open(en_path) as f:
            for line in f:
                line = line.strip()
                word, index = line.split("\t")
                en_dic[word] = index
        with open(cn_path) as f:
            for line in f:
                line = line.strip()
                word, index = line.split("\t")
                cn_dic[word] = index

        with open(trans_dic_path) as f, open(output_path, "w") as fw:
            for line in f:
                line = line.strip()
                ll = line.split("\t")
                if len(ll) != 2:
                    continue
                word_en, word_cn = ll
                index_en = en_dic[word_en]
                if word_cn not in cn_dic.keys():
                    continue
                index_cn = cn_dic[word_cn]
                print("\t".join([index_en, index_cn, word_en, word_cn]), file = fw)
