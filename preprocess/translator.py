# -*- coding: utf-8 -*-

import urllib.request as ur
import json
import random
import hashlib

class translator:
    
    def __init__(self, id, key, base_url, encoding = "utf-8"):
        self.id = id
        self.key = key
        self.base_url = base_url
        self.encoding = encoding
        
    def request(self, string):
        req_dic = {}
        random_int = int(random.random()*10000000)
        req_dic["q"] = string
        req_dic["from"] = "en"
        req_dic["to"] = "zh"
        req_dic["appid"] = self.id
        req_dic["salt"] = str(random_int)
        req_dic["sign"] = self.id + string + str(random_int) + self.key
        req_dic["sign"] = hashlib.md5(req_dic["sign"].encode(self.encoding)).hexdigest()  
        res = ""
        try:
            url = self.base_url + "&".join(["=".join([x, y]) for x, y in req_dic.items()])
            fp = ur.urlopen(url).read()
            fp = str(fp, self.encoding)
            obj = json.loads(fp)
            res = obj['trans_result'][0]['dst']
        except Exception as e:
            print(e)
        return res
    
    def translate(self, origin_path, translate_path):
        with open(origin_path, "r") as f, open(translate_path, "w") as fw:
            for line in f:
                word =  line.strip().split("\t")[0]
                word_trans = self.request(word)
                print("\t".join([word, word_trans]))
                print("\t".join([word, word_trans]), file = fw)
