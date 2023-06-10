from torch.utils.data import Dataset, DataLoader
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, LancasterStemmer
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import math
import torch
import spacy
import random
import json
import codecs
import csv
import string
import jsonlines
import torch.nn.functional as F
from collections import Counter
from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel
os.environ['GENSIM_DATA_DIR'] = './gensim-data'

porter = PorterStemmer()
# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')


def get_prediction(tokenizer, indexed_tokens, keywords_gpt, predicted_index, guarantee, T_time, time):

    if guarantee and time > T_time:
        predicted_index = list(keywords_gpt.keys())[0]
    if guarantee and predicted_index in keywords_gpt:
        predicted_text = tokenizer.decode(
            indexed_tokens) + ' ' + keywords_gpt[predicted_index]
        pred_word = keywords_gpt[predicted_index]
    else:
        # predicted_index = predicted_index[0].item()
        # logger.info(indexed_tokens)
        # logger.info(predicted_index)

        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        # this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
        print("indexed_tokens + [predicted_index]",
              indexed_tokens + [predicted_index])
        print("predicted_text", predicted_text)
        pred_word = predicted_text.split()[-1].split('<|endoftext|>')[-1]
        print("pred_word", pred_word)

    return pred_word, predicted_text, predicted_index,


def distilGPT2_perplexity_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor(
        [tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss, logits = model(tensor_input, labels=tensor_input)[:2]

    return math.exp(loss)


def distinct_n(example, n, n_distinct, n_total, counter):
    """
    Gives the number of distinct n-grams as well as the total n-grams
    Args:
        example: input text
        n: n-grams size (i.e., the n)
        n_distinct: distinct n-grams in previous iteration
        n_total: total n-grams in previous iteration
        counter: token counter in previous iteration, i.e., how many times a token appeared

    """
    for token in zip(*(example[i:] for i in range(n))):
        if token not in counter:
            n_distinct += 1
        elif counter[token] == 1:
            n_distinct -= 1
        counter[token] += 1
        n_total += 1
    if n_total == 0:
        n_total = 1
    return n_distinct, n_total, counter


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove )

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def get_weight(weight, guarantee, T_time, time):

    if guarantee:
        if T_time == 0:
            T_time = 1
        weight = float(weight)
        # 100 is the maximum value the weight will reach
        rate = (1/T_time)*np.log(100/weight)
        weight = weight*np.exp(rate*time)

    return weight


def get_keywordsets(task='commentgen', file_name="/home/blockchain/wanghl/train_lm_with_keywords/data_process/test_title_search_in_train_all.csv"):
    if task == 'commentgen':
        keyword_sets = []
        csv.field_size_limit(500 * 1024 * 1024)
        with codecs.open(file_name, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                # print(row)
                keywords = row['keywords']
                if keywords != '':
                    keywords = list(keywords.split())
                else:
                    keywords = []
                newspara = row['news_paras']
                in_text = newspara
                keyword_sets.append([in_text, keywords])

    else:
        # File containing the keywords as text
        in_text = '<|endoftext|>'  # 'Start with EOS
        # in_texts = ['I', 'It', 'A'] #Other possible start tokens
        file1 = open(file_name, "r+")
        lines = file1.readlines()
        if task == 'commongen':
            keyword_sets = [(in_text, list(line.strip().split()))
                            for line in lines]
        else:
            keyword_sets = [(in_text, list(line.strip().split(", ")))
                            for line in lines]
            # keyword_sets = [(random.choice(in_texts), list(line.strip().split(", "))) for line in lines]
    return keyword_sets


def get_keywordsets_bitstream(task='commentgen', file_name="/data2/yahoo_news_release/test_title_search_in_dev_all.csv"):
    if task == 'commentgen':
        keyword_sets = []
        csv.field_size_limit(500 * 1024 * 1024)
        with codecs.open(file_name, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                # print(row)
                keywords = row['keywords']
                if keywords != '':
                    keywords = list(keywords.split())
                else:
                    keywords = []
                newspara = row['news_paras']
                in_text = newspara

                stext = row['stext']
                # sword_index = row['sword_index'][1:-1].split()
                sword_index = list(row['sword_index'].split())
                news_body = row['news_body']
                # 这里有两种方式转化秘密信息比特流判断最大长度
                # 1种是利用新闻的最大长度/2 1种是利用秘密信息索引值的最大长度/2
                #
                s_index = []

                for index in sword_index:
                    s_index.append(int(index))
                max_index = len(bin(max(s_index)).replace('0b', ''))
                max_len = len(bin(len(news_body.split())).replace('0b', ''))
                # max_len = max(words_index)
                bit_stream = ''
                # zfill可以在前面补0
                for index in sword_index:
                    tmp_bits = bin(
                        int(index)).replace('0b', '').zfill(max_len)
                    bit_stream = bit_stream+tmp_bits

                # s = "".join(f"{num:08o}")  # 指定生成8位
                # num 是你要转换的十进制数字，0我也不知道是啥作用，不要动，8代表你要保证多少位，o代表八进制

                keyword_sets.append([in_text, keywords, bit_stream])

    else:
        # File containing the keywords as text
        in_text = '<|endoftext|>'  # 'Start with EOS
        # in_texts = ['I', 'It', 'A'] #Other possible start tokens
        file1 = open(file_name, "r+")
        lines = file1.readlines()
        if task == 'commongen':
            keyword_sets = [(in_text, list(line.strip().split()))
                            for line in lines]
        else:
            keyword_sets = [(in_text, list(line.strip().split(", ")))
                            for line in lines]
            # keyword_sets = [(random.choice(in_texts), list(line.strip().split(", "))) for line in lines]
    return keyword_sets


def get_keywordsets_bitstream_jsonl_wo_unk(file_name="/data2/yahoo_news_release/test_title_search_in_dev_all.csv", enc_dict={}):

    keyword_sets = []
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
                newspara = row['news_paras']
                in_text = newspara
                stext = row['stext']
                # sword_index = row['sword_index'][1:-1].split()
                sword_index = list(row['sword_index'].split())

                # 这里有两种方式转化秘密信息比特流判断最大长度
                # 1种是利用新闻的最大长度/2 1种是利用秘密信息索引值的最大长度/2
                #
                s_index = []
                for index in sword_index:
                    s_index.append(int(index))
                max_index = len(bin(max(s_index)).replace('0b', ''))
                # max_len = len(bin(len(news_body.split())).replace('0b', ''))
                # max_len = max(words_index)
                bit_stream = ''
                # zfill可以在前面补0
                for index in sword_index:
                    tmp_bits = bin(
                        int(index)).replace('0b', '').zfill(max_index)
                    bit_stream = bit_stream+tmp_bits

                    # s = "".join(f"{num:08o}")  # 指定生成8位
                    # num 是你要转换的十进制数字，0我也不知道是啥作用，不要动，8代表你要保证多少位，o代表八进制

                keyword_sets.append([in_text, keywords, bit_stream])

    return keyword_sets


def get_keywordsets_bitstream_jsonl_wo_unk_v2(file_name="/data2/yahoo_news_release/test_title_search_in_dev_all.csv", enc_dict={}):
    # 这里秘密信息的最开始5bit表示秘密信息index的数量 方便后续解码
    keyword_sets = []
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

                newspara = row['news_paras']
                in_text = newspara
                stext = row['stext']
                # sword_index = row['sword_index'][1:-1].split()
                sword_index = list(row['sword_index'].split())

                # 这里有两种方式转化秘密信息比特流判断最大长度
                # 1种是利用新闻的最大长度/2 1种是利用秘密信息索引值的最大长度/2
                #
                num = len(sword_index)
                num_bit = bin(
                    int(num)).replace('0b', '').zfill(5)
                s_index = []
                for index in sword_index:
                    s_index.append(int(index))
                max_index = len(bin(max(s_index)).replace('0b', ''))
                # max_len = len(bin(len(news_body.split())).replace('0b', ''))
                # max_len = max(words_index)
                bit_stream = num_bit
                # zfill可以在前面补0
                for index in sword_index:
                    tmp_bits = bin(
                        int(index)).replace('0b', '').zfill(max_index)
                    bit_stream = bit_stream+tmp_bits

                    # s = "".join(f"{num:08o}")  # 指定生成8位
                    # num 是你要转换的十进制数字，0我也不知道是啥作用，不要动，8代表你要保证多少位，o代表八进制

                keyword_sets.append([in_text, keywords, bit_stream])

    return keyword_sets


def get_jsonl_wo_unk_v3(file_name="/data2/yahoo_news_release/test_title_search_in_dev_all.jsonl", enc_dict={}):
    # 这里秘密信息的最开始5bit表示秘密信息index的数量 方便后续解码
    with jsonlines.open('/data2/test_title_search_in_train_all_wo_unk.jsonl', "w") as file:
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
                    file.write(row)


def get_keywordsets_bitstream_jsonl(task='commentgen', file_name="/data2/yahoo_news_release/test_title_search_in_dev_all.csv"):
    if task == 'commentgen':
        keyword_sets = []
        with open(file_name, "r", encoding="utf-8") as f:
            for row in jsonlines.Reader(f):
                keywords = row['keywords']
                if keywords != '':
                    keywords = list(keywords.split())
                else:
                    keywords = []
                newspara = row['news_paras']
                in_text = newspara
                stext = row['stext']
                # sword_index = row['sword_index'][1:-1].split()
                sword_index = list(row['sword_index'].split())

                # 这里有两种方式转化秘密信息比特流判断最大长度
                # 1种是利用新闻的最大长度/2 1种是利用秘密信息索引值的最大长度/2
                #
                s_index = []
                for index in sword_index:
                    s_index.append(int(index))
                max_index = len(bin(max(s_index)).replace('0b', ''))
                # max_len = len(bin(len(news_body.split())).replace('0b', ''))
                # max_len = max(words_index)
                bit_stream = ''
                # zfill可以在前面补0
                for index in sword_index:
                    tmp_bits = bin(
                        int(index)).replace('0b', '').zfill(max_index)
                    bit_stream = bit_stream+tmp_bits

                # s = "".join(f"{num:08o}")  # 指定生成8位
                # num 是你要转换的十进制数字，0我也不知道是啥作用，不要动，8代表你要保证多少位，o代表八进制
                keyword_sets.append([in_text, keywords, bit_stream])
    else:
        # File containing the keywords as text
        in_text = '<|endoftext|>'  # 'Start with EOS
        # in_texts = ['I', 'It', 'A'] #Other possible start tokens
        file1 = open(file_name, "r+")
        lines = file1.readlines()
        if task == 'commongen':
            keyword_sets = [(in_text, list(line.strip().split()))
                            for line in lines]
        else:
            keyword_sets = [(in_text, list(line.strip().split(", ")))
                            for line in lines]
            # keyword_sets = [(random.choice(in_texts), list(line.strip().split(", "))) for line in lines]
    return keyword_sets


def glove_encode(glove_encoder, word):
    return glove_encoder(word)


def get_score(k, number_of_words_per_sentence, online_probability, proba):
    alpha = 0.6
    length = (k+1)*number_of_words_per_sentence
    len_norm = ((5+length)**alpha)/(6**alpha)
    score_ = np.log(online_probability*proba)/len_norm

    return score_


def get_logits(model, tokenizer, text, this_sequence, temperature):
    # GPT2 - generate logits
    '''
    在这里把上文加上一个<comment is>
    '''
    # text = text + ' <comment is> '
    indexed_tokens = tokenizer.encode(text)
    indexed_this_seq = tokenizer.encode(this_sequence)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')

    # Predict all tokens
    outputs = model(tokens_tensor)

    del tokens_tensor
    torch.cuda.empty_cache()

    logits = outputs.logits
    logits = logits[0, -1, :] / temperature

    return logits, indexed_tokens, indexed_this_seq


def distilGPT2_perplexity_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor(
        [tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss, logits = model(tensor_input, labels=tensor_input)[:2]

    return math.exp(loss)


def get_sim(keywords_enc, keywords_gpt, converter_table, guarantee, mode, only_max):
    converter_table_tensor = torch.FloatTensor(converter_table)
    converter_table_tensor = converter_table_tensor.to('cuda')

    if len(keywords_enc) > 1:
        # sims_tensor = torch.empty_like(keywords_enc[0])
        sims_tensor = []
        for w in keywords_enc:
            w = np.reshape(w, (1, -1))
            w_tensor = torch.FloatTensor(w)
            w_tensor = w_tensor.to('cuda')
            sims_tmp = F.cosine_similarity(w_tensor, converter_table_tensor)
            sims_tensor.append(sims_tmp)

        # torch.cat(sims_tmp)
        # sims = torch.unsqueeze(sims_tensor, dim=1)
        sims = torch.stack(sims_tensor)
        sims = torch.unsqueeze(sims, dim=1)
        '''
        sims = np.array([sims_tmp.cpu().numpy().reshape(1, -1)
                         for sims_tmp in sims_tensor])
        '''
        del converter_table_tensor
        del sims_tensor
        del w_tensor
        torch.cuda.empty_cache()
        if guarantee:
            for i, w in enumerate(keywords_gpt):
                sims[i][0][w] = 1
        if mode == 'max':
            # sim = np.max(sims, axis=0)
            sim = torch.max(sims, axis=0)[0]
        elif mode == 'all':
            # sim = np.sum(sims, axis=0)
            sim = torch.sum(sims, axis=0)
        else:
            raise Exception(
                "keywords_enc length is greater than 1 so expect to be in mode 'max' or 'all'")
    else:
        keywords_enc[0] = np.reshape(keywords_enc[0], (1, -1))
        keywords_enc_tensor = torch.FloatTensor(keywords_enc[0])
        keywords_enc_tensor = keywords_enc_tensor.to('cuda')
        sim = F.cosine_similarity(keywords_enc_tensor, converter_table_tensor)
        # sim = torch.unsqueeze(sim, dim=1)
        '''
        sim = cosine_similarity(np.reshape(
            keywords_enc[0], (1, -1)), converter_table)
        '''
    # Only the target word, not the neighbour (as measured by cosine similarity)
    if only_max == True:
        sim_aux = np.zeros_like(sim)
        sim_aux[0, sim.argmax()] = sim.max()
        sim = np.squeeze(sim_aux)
    else:
        # sim = np.clip(np.squeeze(sim), a_min=0, a_max=None)  # tf.square(sim)
        sim = torch.clip(torch.squeeze(sim), min=0, max=None)

    return sim


def get_keywords(keywords, enc_dict, tokenizer, mode):
    keywords_ = [w for w in keywords]

    # Select the next guide word(s)
    if keywords_:
        if mode == 'next':
            keywords_ = [keywords_[0]]
        if mode == 'random':
            keywords_ = [random.choice(keywords_)]

    keywords_enc = [enc_dict[w] for w in keywords_]
    keywords_gpt = {tokenizer.encode(w)[0]: w for w in keywords_}

    return keywords_enc, keywords_gpt


def checker(string):
    string = string.replace("'ve", '')
    string = string.replace("@", '')
    string = string.replace("'re", '')
    string = string.replace("'d", '')
    string = string.replace("?", '')
    string = string.replace("'s", '')
    string = string.replace(":", '')
    string = string.replace("!", '')
    string = string.replace('"', '')
    string = string.replace(".", '')
    string = string.replace("--", '')
    string = string.replace("'", '')
    string = string.replace(",", '')
    string = string.replace(';', '')
    string = string.replace('‘', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('\'', '')
    string = string.replace(' ', '')
    return(string)


# Pytorch
def converter_table_glove():
    import gensim.downloader as api
    glove_encoder = api.load("glove-wiki-gigaword-300")

    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/data/converter_table_glove'

    # load gpt-2 model
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model.eval()

    holder = np.zeros((50257, 300))

    # translate every word from the gpt-2 space into a glove representation
    for i in range(50257):
        try:
            word = tokenizer.decode([i])
            word = checker(word.strip().lower())
            glove = glove_encoder[word]
            holder[i, :] = glove
        except:
            word = tokenizer.decode([i])
            holder[i, :] = np.zeros((300))  # + 500

    # Save all 50'000 glove representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Table was generated')


def converter_table_word2vec():
    import gensim.downloader as api
    word2vec_encoder = api.load("word2vec-google-news-300")

    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/data/converter_table_word2vec'

    # load gpt-2 model
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model.eval()

    holder = np.zeros((50257, 300))

    # translate every word from the gpt-2 space into a word2vec representation
    for i in range(50257):
        try:
            word = tokenizer.decode([i])
            word = checker(word.strip().lower())
            word2vec = word2vec_encoder[word]
            holder[i, :] = word2vec
        except:
            word = tokenizer.decode([i])
            holder[i, :] = np.zeros((300))  # + 500

    # Save all 50'000 word2vec representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Table was generated')


def count_word_stem_one(word, sequence):
    # print ("Sequence", sequence)
    sequence = sequence.split()

    word_count = 0
    word_stem = porter.stem(word.lower())
    for s_word in sequence:
        s_word_stem = porter.stem(s_word.lower())
        if(s_word_stem == word_stem):
            word_count = 1
            break

    return word_count


def count_word_stem(word, sequence):
    # print ("Sequence", sequence)
    sequence = sequence.split()
    word_count = 0

    word_stem = porter.stem(word.lower())

    for s_word in sequence:
        s_word_stem = porter.stem(s_word.lower())
        # print(s_word_stem)
        if(s_word_stem == word_stem):
            word_count += 1

    return word_count

# A score function for the quality of the sentence


def evaluate_quality(sequence, word, related_count, perplexity, guide, temp=1.):
    # we aim for one ocurance of the word,  and low perplexity
    w_1 = 1
    w_3 = 0.001
    c_star = 2

    if(word == ""):
        quality_score = math.exp(-(w_1*(c_star) + w_3*perplexity))
        return quality_score

    quality_score = 0
    word_count = count_word_stem(word, sequence)

    if(word_count != 0) and guide:
        quality_score = math.exp(-(w_1*word_count + w_3*perplexity))
    else:
        quality_score = math.exp(-(w_1*(c_star) + w_3*perplexity))

    quality_score = quality_score/temp
    # DEBUG
    # print("txt, quality_score, word_count, rel_count, ppl", sequence, quality_score, word_count, related_count, perplexity)

    return quality_score, word_count


# A score function for the quality of the sentence
def evaluate_quality_linear(sequence, word_count, perplexity, temp=1., perp=False):
    # we aim for one ocurance of the word,  and low perplexity
    w_1 = 1
    w_3 = 0.01

    if perp:
        quality_score = word_count - w_3*perplexity
    else:
        quality_score = word_count + w_3*perplexity

    quality_score = quality_score/temp  # Temperature for sampling

    return quality_score


# simply the negative cosine similarity for use in calculating the 'best_tour'
def neg_cosine_similarity(v, w):
    return -1 * cosine_similarity(np.reshape(v, (1, -1)), np.reshape(w, (1, -1)))

# simply the positive cosine similarity for use in calculating the "worst" tour using 'best_tour' - used only as a sanity check (is worst worse than best?)


def pos_cosine_similarity(v, w):
    return cosine_similarity(np.reshape(v, (1, -1)), np.reshape(w, (1, -1)))

# function to calculate the total tour length when visiting the words in the given 'order'


def tour_length(distance_matrix, order):
    length = 0
    for i, j in zip(order, order[1:]):
        length += distance_matrix[i][j]
    return length

# find the best tour through the guide words, minimizing the pairwise distance between consecutive guide words


def best_tour(glove_array, distance=neg_cosine_similarity, top_k=1):
    """
    returns the best order to minimize the tour length
    default pairwise distance is the negative cosine similarity
    input should be an nxm array of word embeddings, where n is no. words and m is length of the word embedding
    *NOT IMPLEMENTED: set top_k to the beam length if you want to use a separate order per beam.
    """
    n = len(glove_array)
    distance_matrix = np.zeros((n, n))
    for i, v in enumerate(glove_array):
        for j, w in enumerate(glove_array):
            distance_matrix[i][j] = distance(v, w)
    tours = {}
    for i in itertools.permutations(list(range(n))):
        tours[i] = tour_length(distance_matrix, i)
        best_tour = min(tours, key=tours.get)
    return best_tour


class KeywordsDataset(Dataset):
    """Keywords dataset."""

    def __init__(self, keyword_sets):
        self.keyword_sets = keyword_sets

    def __len__(self):
        return len(self.keyword_sets)

    def __getitem__(self, idx):
        return self.keyword_sets[idx]


if __name__ == '__main__':
    get_keywordsets_bitstream()
