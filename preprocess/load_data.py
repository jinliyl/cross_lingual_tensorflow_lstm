# -*- coding: utf-8 -*-
import os
import re

class load_data():

    #load original file and make it standard
    def __init__(self, dataset_name, output_path, lang, comment_path, label_path = "", encoding = "utf-8"):
        
        self.dataset_name = dataset_name
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        self.lang = lang
        self.comment_path = comment_path
        self.label_path = label_path
        self.encoding = encoding

        self.label_list = []
        self.comment_list = []
        self.word_dic = {}


    def clean_str(self, string):
        # string replace
        if self.lang == "en":
            string = re.sub("&gt;", " ", string)
            string = re.sub("&lt;", " ", string)
            string = re.sub("&amp;", " & ", string)
            string = re.sub(r"[^A-Za-z0-9(),<>&!?\'\"\`]", " ", string)
            string = re.sub(r"\'s", " \'s ", string)
            string = re.sub(r"\'ve", " \'ve ", string)
            string = re.sub(r"n\'t", " n\'t ", string)
            string = re.sub(r"\'re", " \'re ", string)
            string = re.sub(r"\'d", " \'d ", string)
            string = re.sub(r"\'ll", " \'ll ", string)
            string = re.sub(r"\t", " ", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " ( ", string)
            string = re.sub(r"\)", " ) ", string)
            string = re.sub(r"\?", " ? ", string)
            string = re.sub(r"\"", " \" ", string)
            string = re.sub(r"\'\'", " \'\' ", string)
            string = re.sub(r"\s{2,}", " ", string) # white space
        else:
            string = re.sub("& gt ;","", string)
            string = re.sub("& lt ;","", string)
            string = re.sub("& quot ;", "", string)
            string = re.sub("& # 44 ;", "", string)

        res_array = []
        #special process
        if self.lang == "en":
            string_array = string.strip().lower().split(" ")
            for _str in string_array:
                if "\'" in _str and len(_str) > 3:
                    for _tmp in re.sub("\'"," \' ", _str).split(" "):
                        if _tmp:
                            if _tmp[0] in [str(x) for x in range(10)]:
                                res_array.append("<>")
                            else:
                                res_array.append(_tmp)
                else:
                    if _str:
                        if _str[0] in [str(x) for x in range(10)]:
                            res_array.append("<>")
                        else:
                            res_array.append(_str)
        else:
            for _str in string.split(" "):
                _str = _str.strip()
                if _str:
                    #if _str[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '@',
                    if _str[0] in ['-', '@',
                        '=', 'ｙ', 'ｏ', 'ｎ', 'Ｓ', 'Ｊ', 'Ｂ', '８', '３', '２', '１', '﹫']:
                        pass
                    else:
                        res_array.append(_str)
        return res_array

    def load_dataset(self):
        print("start loading dataset " + self.dataset_name + " ...")
        if self.lang == "en":
            
            comment_path_list = []
            for _domain in os.listdir(self.comment_path):
                if os.path.isdir('/'.join([self.comment_path, _domain])):
                    for _file in os.listdir('/'.join([self.comment_path, _domain])):
                        comment_path_list.append('/'.join([self.comment_path, _domain, _file]))

            for _path in comment_path_list:
                with open(_path, "r") as f:
                    for line in f:
                        ll = list(map(lambda x:x.strip(), line.strip().split('\t')))
                        if len(ll) == 3 and ll[1] not in ['-', '']:
                            self.label_list.append(int(ll[1]))
                            self.comment_list.append(self.clean_str(ll[2]))
        else:
            with open(self.label_path) as f:
                for line in f:
                    line = line.strip()
                    self.label_list.append(int(line))
            with open(self.comment_path) as f:
                for line in f:
                    line = line.strip()
                    self.comment_list.append(self.clean_str(line))
        print("load compleate...")


    def save_dataset(self):
        print("start saving dataset " + self.dataset_name + " ...")
        output_path_name = "/".join([self.output_path, self.dataset_name])
        with open(output_path_name, "w") as f:
            for label, comment in zip(self.label_list, self.comment_list):
                for _c in comment:
                    #if _c in self.word_dic.keys():
                    #    self.word_dic[_c] += 1
                    #else:
                    #    self.word_dic[_c] = 1
                    if _c not in self.word_dic.keys():
                        self.word_dic[_c] = len(self.word_dic)
                print("\t".join([str(label), " ".join(comment)]), file = f)
        
        with open(output_path_name + "_dic", "w") as f:
            word_list = sorted(self.word_dic.items(), key = lambda x:x[1], reverse = False)
            for word, count in word_list:
                print("\t".join([word, str(count)]), file = f)
        print("dataset saved...")

