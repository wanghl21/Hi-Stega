# 计算隐写文本中到底包含了多少秘密信息
import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse
import csv
import codecs
import jsonlines
import gensim.downloader as api
import pickle
import argparse

word_embedding = {
    'glove': "glove-wiki-gigaword-300",
    'word2vec': "word2vec-google-news-300"
}


def get_keywordsets_bitstream_jsonl_wo_unk_v2(file_name="/data2/yahoo_news_release/test_title_search_in_dev_all.csv", enc_dict={}):
    # 这里秘密信息的最开始5bit表示秘密信息index的数量 方便后续解码
    keyword_sets = []
    all_stext = []
    with open(file_name, "r", encoding="utf-8") as f:
        for row in jsonlines.Reader(f):
            keywords = row['keywords']
            if keywords != '':
                keywords = list(keywords.split())
            else:
                keywords = []
            flag = 1
            for keyword in keywords:
                if keyword not in enc_dict.keys():
                    flag = 0
                    print(keyword)
            if flag:
                stext = row['stext']
                all_stext.append(stext)

    return all_stext


embedding = "glove"
file_name = "/data2/yahoo_news_release/test_title_search_in_train_all.jsonl"
save_path_dict = "/home/blockchain/wanghl/train_lm_with_keywords/data/dict_wo_unktest_title_search_in_train_allglove.pkl"
with open(save_path_dict, 'rb') as file:
    enc_dict = pickle.load(file)
all_bit_nums = 0
all_text = get_keywordsets_bitstream_jsonl_wo_unk_v2(
    file_name=file_name, enc_dict=enc_dict)
bit_num = 0
all_letter_num = 0
for tmp in all_text:
    words = tmp.split(" ")
    for word in words:
        all_letter_num += len(word)
print(all_letter_num)
all_bit_nums = all_letter_num*8
print(all_bit_nums)
all_stego_length = 0
stego_file = '/home/blockchain/wanghl/train_lm_with_keywords/result_randomseed/generate_keyword/max_ac_gpt2_train_all_2022.11.22-11:03:49_5.0.jsonl'
with open(stego_file, "r", encoding="utf-8") as sf:
    for row in jsonlines.Reader(sf):
        if 'stego' in row.keys():
            stgeo = row['stego']
            all_stego_length += len(stgeo.split(" "))
print(all_stego_length)

print(all_bit_nums/all_stego_length)
