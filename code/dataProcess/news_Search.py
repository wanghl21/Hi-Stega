import json
import string
import torch
import utils
import random
import datetime
import csv
import codecs
from TweetClean import *
import jsonlines
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
csv.field_size_limit(500 * 1024 * 1024)


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def is_english(strs):
    import string
    strs = strQ2B(strs)
    # 祛除了‘“”
    for idx in range(len(strs)):
        i = strs[idx]
        if ord(i) > 127:
            return False
    return True


def context_clean(paras):

    paras = paras.lower()
    paras = paras.translate(str.maketrans('', '', string.punctuation))
    paras = [word for word in paras.split() if word.isalpha()]
    return paras


def rewrite(datafile, fileName):
    data_file = open(datafile,
                     'r', encoding='utf-8')
    with open(fileName, 'w', encoding='utf-8')as file:

        for l in data_file.readlines():
            line = json.loads(l)
            paras = ''
            for para in line['paras']:
                paras = paras + para.strip()
            if (is_english(paras[0:20])):
                file.writelines(l)


def random_title(randomsample=True, seed=2, datafile='/data2/yahoo_news_release/test.data', fileName='test_title.txt'):
    data_file = open(datafile,
                     'r', encoding='utf-8')
    titles = []
    final_titles = []
    for l in data_file.readlines():
        line = json.loads(l)
        title = line['title']
        # print(title)
        titles.append(title)
    if randomsample:
        random.seed(seed)
        r2 = random.sample(range(1, 2000), 10)
        with open(fileName, 'w', encoding='utf-8')as file:
            for i in r2:
                file.write(titles[i]+'\n')
                final_titles.append(titles[i])
        return final_titles
    else:
        with open(fileName, 'w+', encoding='utf-8')as file:
            for tmp in titles:
                title = tmp
                file.write(title+'\n')
                final_titles.append(title)
        return final_titles


def random_comment(randomsample=True, seed=2, datafile='/data2/yahoo_news_release/test.data', fileName='test_comment.txt'):
    data_file = open(datafile,
                     'r', encoding='utf-8')
    comments = []
    final_comments = []

    for l in data_file.readlines():
        line = json.loads(l)
        max_vote = 0
        for tmp in line['cmts']:
            if tmp['upvote']+tmp['downvote'] > max_vote:
                cmts = tmp['cmt']
                max_vote = tmp['upvote']+tmp['downvote']
        comments.append(cmts)
    if random:
        random.seed(seed)
        r2 = random.sample(range(1, 2000), 100)
        with open(fileName, 'w', encoding='utf-8')as file:
            for i in r2:
                file.write(comments[i]+'\n')
                final_comments.append(comments[i])
        return final_comments
    else:
        with open(fileName, 'w', encoding='utf-8')as file:
            for tmp in comments:
                file.write(tmp+'\n')
        return comments


def get_stext(fileName='test_comment.txt'):
    text = []
    with open(fileName, 'w', encoding='utf-8')as file:
        for l in file.readlines():
            text.append(l)


def search_news(text):
    with open('test_title_search_in_train_all.csv'+log_stamp.strftime('%Y.%m.%d-%H:%M:%S'), 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["stext", "keywords_len", "keywords", "final_keywords_num", "news_paras", "news_cmts", "news_body", "newsline"])
        for tmptext in text[29:]:
            # stext = context_clean(tmptext)
            stext = tweet_cleaning_for_sentiment_analysis(
                tmptext, contain_punc=False)
            stext = stext.translate(str.maketrans('', '', string.punctuation))
            stext = stext.split()
            print("stext:", stext)
            print("len(stext)", len(stext))
            max_count = 0
            final_paras = ''
            final_line = ''
            keywords = []
            data_file = open("/data2/yahoo_news_release/train.data",
                             'r', encoding='utf-8')
            for l in data_file.readlines():
                line = json.loads(l)
                paras = ''
                for para in line['paras']:
                    paras = paras + para.strip()
                # paras = context_clean(paras)
                paras = tweet_cleaning_for_sentiment_analysis(
                    paras, contain_punc=False)
                paras = paras.translate(str.maketrans(
                    '', '', string.punctuation)).split()
                count = 0
                keywords = []
                for wd in stext:
                    if wd in paras:

                        count = count+1
                    else:
                        keywords.append(wd)
                print(count)
                if count > max_count:
                    max_count = count
                    final_paras = paras
                    final_line = line
                    final_keywords = keywords
                    print(max_count)
                    print(keywords)

                if max_count == len(stext):
                    break
            print("final_keywords", final_keywords)
            final = ' '.join(final_keywords)
            print(final_line['paras'][0])
            tokenize_input = tokenizer.tokenize(final_line['paras'][0])
            # 处理new body 选择news的主要信息
            tmp_paras = ''
            if len(tokenize_input) <= 10 and len(final_line['paras']) >= 2:
                if 'By' in final_line['paras'][0] or 'From' in final_line['paras'][0]:
                    tmp_paras = final_line['paras'][1]
                else:
                    tmp_paras = final_line['paras'][0]+final_line['paras'][1]
                    # print("line['paras'][0]:   ", line['paras'][0])
            else:
                tmp_paras = final_line['paras'][0]
            news_paras = tweet_cleaning_for_sentiment_analysis(
                tmp_paras, contain_punc=False)
            sum_vote = 0
            # 选择vote最多的cmts
            for tmp in final_line['cmts']:
                if tmp['upvote']+tmp['downvote'] > sum_vote:
                    tmp_cmts = tmp['cmt']
                    sum_vote = tmp['upvote']+tmp['downvote']
            news_cmts = tweet_cleaning_for_sentiment_analysis(
                tmp_cmts, contain_punc=False)
            writer.writerow(
                [' '.join(stext), len(stext), final, len(final_keywords), news_paras, news_cmts, final_paras, final_line])


def search_news_fast(text):
    clean_paras_list = []
    lines_list = []
    if not os.path.exists("/data2/yahoo_news_release/train_data_clean_for_train_all.csv"):
        data_file = open("/data2/yahoo_news_release/train.data",
                         'r', encoding='utf-8')
        with open('/data2/yahoo_news_release/train_data_clean_for_train_all.csv', 'w+', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["clean_paras", "newsline"])

            for l in data_file.readlines():
                line = json.loads(l)
                paras = ''
                for para in line['paras']:
                    paras = paras + para.strip()
                    # paras = context_clean(paras)
                paras = tweet_cleaning_for_sentiment_analysis(
                    paras, contain_punc=False)
                paras = paras.translate(str.maketrans(
                    '', '', string.punctuation))
                writer.writerow([paras, line])
                clean_paras_list.append(paras)
                lines_list.append(line)
    else:
        csv_file = '/data2/yahoo_news_release/train_data_clean_for_train_all.csv'
        # csv_file = '/data2/yahoo_news_release/dev_data_clean_for_dev_all.csv'
        with codecs.open(csv_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                clean_paras_list.append(row['clean_paras'])
                lines_list.append(eval(row['newsline']))

                # print(row)

    with open('/data2/yahoo_news_release/test_title_search_in_train_all.csv', 'w+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["stext", "keywords_len", "keywords", "final_keywords_num", "sword_index", "news_paras", "news_cmts", "news_body"])

        for tmptext in text:
            # stext = context_clean(tmptext)
            stext = tweet_cleaning_for_sentiment_analysis(
                tmptext, contain_punc=False)
            stext = stext.translate(str.maketrans('', '', string.punctuation))
            stext = stext.split()
            if len(stext) > 1:
                print("stext:", stext)
                print("len(stext)", len(stext))
                max_count = 0
                final_paras = ''
                final_line = ''
                final_sword_index = []
                for index in range(len(clean_paras_list)):
                    paras = clean_paras_list[index]
                    # print(paras)
                    count = 0
                    sword_index = []
                    keywords = []
                    for wd in stext:
                        if wd in paras.split():
                            # if wd in paras:
                            wd_index = paras.split().index(wd)
                            sword_index.append(str(wd_index))
                            count = count+1

                        else:
                            keywords.append(wd)
                    # print(count)
                    if count > max_count:
                        max_count = count
                        final_paras = paras
                        final_line = lines_list[index]
                        final_keywords = keywords
                        final_sword_index = sword_index
                        print(max_count)
                        print(keywords)
                        print(final_sword_index)
                        # print(paras)
                        # print(line)
                    if max_count == len(stext):
                        break
                print("final_keywords", final_keywords)
                final = ' '.join(final_keywords)
                print(final_line['paras'][0])
                tokenize_input = tokenizer.tokenize(final_line['paras'][0])
                # 处理new body 选择news的主要信息
                tmp_paras = ''
                if len(tokenize_input) <= 10 and len(final_line['paras']) >= 2:
                    if 'By' in final_line['paras'][0] or 'From' in final_line['paras'][0]:
                        tmp_paras = final_line['paras'][1]
                    else:
                        tmp_paras = final_line['paras'][0] + \
                            final_line['paras'][1]
                        # print("line['paras'][0]:   ", line['paras'][0])
                else:
                    tmp_paras = final_line['paras'][0]
                news_paras = tweet_cleaning_for_sentiment_analysis(
                    tmp_paras, contain_punc=False)
                sum_vote = 0
                # 选择vote最多的cmts
                for tmp in final_line['cmts']:
                    if tmp['upvote']+tmp['downvote'] > sum_vote:
                        tmp_cmts = tmp['cmt']
                        sum_vote = tmp['upvote']+tmp['downvote']
                news_cmts = tweet_cleaning_for_sentiment_analysis(
                    tmp_cmts, contain_punc=False)

                writer.writerow(
                    [' '.join(stext), len(stext), final, len(final_keywords), ' '.join(final_sword_index), news_paras, news_cmts, final_paras])


def search_news_fast_json(text):
    clean_paras_list = []
    lines_list = []
    if not os.path.exists("/data2/yahoo_news_release/train_data_clean_for_train_all.csv"):
        data_file = open("/data2/yahoo_news_release/train.data",
                         'r', encoding='utf-8')
        with open('/data2/yahoo_news_release/train_data_clean_for_train_all.csv', 'w+', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["clean_paras", "newsline"])

            for l in data_file.readlines():
                line = json.loads(l)
                paras = ''
                for para in line['paras']:
                    paras = paras + para.strip()
                    # paras = context_clean(paras)
                paras = tweet_cleaning_for_sentiment_analysis(
                    paras, contain_punc=False)
                paras = paras.translate(str.maketrans(
                    '', '', string.punctuation))
                writer.writerow([paras, line])
                clean_paras_list.append(paras)
                lines_list.append(line)
    else:
        csv_file = '/data2/yahoo_news_release/train_data_clean_for_train_all.csv'
        # csv_file = '/data2/yahoo_news_release/dev_data_clean_for_dev_all.csv'
        with codecs.open(csv_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                clean_paras_list.append(row['clean_paras'])
                lines_list.append(eval(row['newsline']))

    with jsonlines.open('test_title_search_in_train_all.jsonl', "w") as f:
        for tmptext in text:

            stext = tweet_cleaning_for_sentiment_analysis(
                tmptext, contain_punc=False)
            stext = stext.translate(str.maketrans('', '', string.punctuation))
            stext = stext.split()
            if len(stext) > 1:
                print("stext:", stext)
                print("len(stext)", len(stext))
                max_count = 0
                final_paras = ''
                final_line = ''
                final_sword_index = []
                for index in range(len(clean_paras_list)):
                    paras = clean_paras_list[index]
                    # print(paras)
                    count = 0
                    sword_index = []
                    keywords = []
                    for wd in stext:
                        if wd in paras.split():
                            count = count+1

                        else:
                            keywords.append(wd)

                    if count > max_count:
                        for wd in stext:
                            if wd in paras.split():

                                wd_index = paras.split().index(wd)
                                sword_index.append(str(wd_index))
                        max_count = count
                        final_paras = paras
                        final_line = lines_list[index]
                        final_keywords = keywords
                        final_sword_index = sword_index
                        print(max_count)
                        print(keywords)
                        print(final_sword_index)

                    if max_count == len(stext):
                        break
                print("final_keywords", final_keywords)
                final = ' '.join(final_keywords)
                print(final_line['paras'][0])
                tokenize_input = tokenizer.tokenize(final_line['paras'][0])

                tmp_paras = ''
                if len(tokenize_input) <= 10 and len(final_line['paras']) >= 2:
                    if 'By' in final_line['paras'][0] or 'From' in final_line['paras'][0]:
                        tmp_paras = final_line['paras'][1]
                    else:
                        tmp_paras = final_line['paras'][0] + \
                            final_line['paras'][1]
                else:
                    tmp_paras = final_line['paras'][0]
                news_paras = tweet_cleaning_for_sentiment_analysis(
                    tmp_paras, contain_punc=False)
                sum_vote = 0
                # 选择vote最多的cmts
                for tmp in final_line['cmts']:
                    if tmp['upvote']+tmp['downvote'] > sum_vote:
                        tmp_cmts = tmp['cmt']
                        sum_vote = tmp['upvote']+tmp['downvote']
                news_cmts = tweet_cleaning_for_sentiment_analysis(
                    tmp_cmts, contain_punc=False)
                f.write({"stext": ' '.join(stext),
                         "stext_len": len(stext),
                         "keywords": final,
                         "final_keywords_num": len(final_keywords),
                         "sword_index": ' '.join(final_sword_index),
                         "news_paras": news_paras,
                         "news_cmts": news_cmts,
                         })


log_stamp = datetime.datetime.now()
# random_comment(randomsample=False, fileName='random_comment.txt')
text = random_title(randomsample=False, fileName='random_title_test_all.txt')
search_news_fast_json(text=text)
