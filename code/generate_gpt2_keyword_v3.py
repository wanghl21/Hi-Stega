import os
import argparse
import logging
import torch
import datetime
import random
import math
import jsonlines
import utils
from utils import *
from encode_keywords import *
from utility_gpt import *
import warnings

from configparser import ConfigParser
from encode import encoder, encoder_configs
from decode import decoder, decoder_configs
from nltk.stem import PorterStemmer, LancasterStemmer
porter = PorterStemmer()

warnings.filterwarnings("ignore")


def parse_arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Train or Generate",
                        choices=["T", "G", "E"], default="G")
    parser.add_argument("--config_path", type=str,
                        default="generate_configs/Yahoo-gpt-generate_keyword_prompt_adgv2.ini")
    # Yahoo-gpt-generate_wo_prompt.ini
    # Yahoo-gpt-generate.ini
    # gpt-generate.ini
    parser.add_argument("--gpuid", type=str, default="1")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--out_dir", type=str,
                        default="/data/lastness/out-1122")
    return parser.parse_args()


args = parse_arg_main()
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configs = ConfigParser()
configs.read(args.config_path)
log_stamp = datetime.datetime.now()
alg = configs.get("generate", "alg")
model_type = configs.get("model", "model_type")
generatefile = configs.get("model", "model_name_or_path")

if len(generatefile) > 4:
    generatefile = generatefile.split("/")[-2]
mode = args.mode
logger = logging.getLogger(__name__)
out_logger = mode+"_"+alg+"_"+generatefile + \
    "_" + log_stamp.strftime('%Y.%m.%d-%H:%M:%S')
logging.basicConfig(
    filename='log_prompt/' + out_logger + '.log',
    filemode='w+',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(configs)


def generate(args, configs):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # print(trainfile)
    # 一些用来生成关键词生成的参数
    init_weight = configs.get("generate", "weight")
    embedding = configs.get("generate", "embedding")
    guide = configs.get("generate", "guide")
    guarantee = configs.get("generate", "do_guarantee")
    mode = configs.get("generate", "mode")
    only_max = configs.get("generate", "only_max")
    converter_table = np.load(str(os.path.dirname(
        os.path.abspath(__file__))) + '/data/converter_table_' + str(embedding) + '.npy')
    if model_type.lower() == "gpt2":
        out_dir = configs.get("model", "model_name_or_path")
        from transformers import AutoTokenizer, GPT2LMHeadModel
        tokenizer = AutoTokenizer.from_pretrained(out_dir)
        model = GPT2LMHeadModel.from_pretrained(out_dir)
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total params: {:d}".format(total_params))
        total_trainable_params = sum(p.numel()
                                     for p in model.parameters() if p.requires_grad)
        logger.info("Trainable params: {:d}".format(total_trainable_params))

        max_length = configs.getint("generate", "max_length")
        generate_num = configs.getint("generate", "generate_num")
        alg = configs.get("generate", "alg")
        reference_filepath = configs.get("generate", "reference_filepath")
        # reference_filepath = "/data2/yahoo_news_release/test_title_search_in_dev_all.csv"
        kwargs = encoder_configs(configs, alg)

        outfile = mode+"_"+alg+"_"+generatefile + "_" + init_weight
        prompt = configs.get("gpt2", "prompt")
        model.eval()

        with torch.no_grad():
            stega_text = []
            stega_idx = 0
            with jsonlines.open(os.path.join("generate_keyword+num_bit_len_bit", outfile + ".jsonl"), "w") as file:
                # with jsonlines.open(outfile + ".jsonl", "w") as file:
                stega_idx = 0
                all_words_num = 0
                all_bits_num = 0
                all_GPT2_ppl = 0
                all_distilGPT2_ppl = 0
                all_distinct_2 = 0
                all_distinct_3 = 0
                all_distinct_4 = 0

                '''
                secret_sets = get_keywordsets_bitstream(
                    file_name=reference_filepath)  # 得到关键词列表和前文
                '''

                # Create file containing the keyword embeddings
                save_path_dict = os.path.join(
                    "data", 'dict_wo_unk' + reference_filepath.split("/")[-1].split(".")[-2] + embedding + '.pkl')
                if not os.path.isfile(save_path_dict):
                    create_enc_dict_jsonl_wo_unk(
                        reference_filepath, embedding)
                with open(save_path_dict, 'rb') as f:
                    enc_dict = pickle.load(f)
                secret_sets = get_keywordsets_bitstream_jsonl_wo_unk_v3(
                    file_name=reference_filepath, enc_dict=enc_dict)  # 得到关键词列表和前文
                # get_jsonl_wo_unk_v3(file_name=reference_filepath, enc_dict=enc_dict)  # 得到清除unk的文件
               # while len(stega_text) < generate_num:
                for j, keyword_set in enumerate(secret_sets):
                    in_text, keywords, bit_stream_ori = keyword_set
                    keywords_ori = keywords
                    bit_index = 0
                    words_num = 0
                    bits_num = 0
                    failure = 0
                    '''
                    with open(configs.get("generate", "bit_filepath"), 'r', encoding='utf8') as f:
                        bit_stream_ori = f.read().strip()
                        bit_stream = list(bit_stream_ori)
                        bit_stream = ''.join(bit_stream)
                        bit_stream = ""
                        bit_index = int(torch.randint(
                            0, high=100, size=(1,)))
                    '''
                    # keywords = list(keywords.split())
                    # 因为是非定长编码,所以需要在秘密信息比特流后面加一定长度的信息
                    if len(bit_stream_ori) <= max_length * math.log2(tokenizer.vocab_size):
                        bit_stream_shuffle = list(bit_stream_ori)
                        random.shuffle(bit_stream_shuffle)
                        bit_stream = bit_stream_ori + \
                            "".join(bit_stream_shuffle)
                    # bit_stream = bit_stream_ori
                    # sample start word
                    stega_sentence = []
                    # TODO begin
                    prefix = in_text  # 前缀
                    prompt_text = prompt  # 提示词
                    if len(in_text.split()) > 512:
                        prefix = (" ").join(in_text.split()[0:64])  # 前缀过长的话就缩短
                    # prefix = ""
                    print(keywords)
                    print(bit_stream_ori)
                    # print(len(prefix.split()))
                    # print(prefix)
                    encoded_prompt = tokenizer.encode(tokenizer.bos_token + prefix + prompt_text,
                                                      add_special_tokens=False,
                                                      return_tensors="pt")
                    encoded_prompt = encoded_prompt.to(device)
                    input_ids = encoded_prompt
                    stega_bit = []
                    logits = model(input_ids).logits[:, -1, :]
                    logits -= logits.max()
                    probs = torch.exp(logits)
                    for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
                        probs[:, forbidden_id] = 0
                    for forbidden_id in range(256):
                        probs[:, forbidden_id] = 0
                    samp = torch.multinomial(probs, 1)  # 随机选第一个词
                    stega_sentence.append(int(samp.view(1, 1)))
                    x = torch.cat([input_ids, samp], dim=1)
                    if alg.lower() == "ac":
                        # num of intervals
                        max_val = 2 ** kwargs["precision"]
                        # max_val = 2**52
                        cur_interval = [0, max_val]
                        # 前面都是一些预处理 作用是找到第一个单词
                    c_time = 0  # current_time

                    # all_time = max_length - 1  # total_time
                    t_time = max_length-1  # total_time

                    for i in range(max_length - 1):
                        # print(keywords)
                        # print(tokenizer.decode(stega_sentence))
                        if tokenizer.eos_token_id in stega_sentence and bit_index >= len(bit_stream_ori) and len(keyword_set) == 0:
                            print("break1")
                            break
                            # conditional probability distribution
                            # todo begin
                            # 用于采样的
                        logits = model(x).logits[:, -1, :]
                        log_prob = logits
                        log_prob -= log_prob.max()
                        prob = torch.exp(log_prob).reshape(-1)

                        # 用于关键字计算的
                        k_prob = F.log_softmax(logits, dim=-1)

                        for forbidden_id in range(256):
                            # k_prob[forbidden_id] = 0
                            prob[forbidden_id] = 0

                        # todo end
                        prob = prob / prob.sum()  # 得到下一个词的条件概率分布
                        if len(keywords) == 0:  # 关键词嵌入结束了
                            if bit_index > len(bit_stream_ori):  # 秘密信息也嵌入结束了
                                print("bits and words all done!")
                                # 就直接随机选择1个单词输出 不再嵌入秘密信息
                                # samp = torch.multinomial(prob, 1)

                                # 或者选择top1输出
                                samp = torch.topk(prob, 1)[1]
                                prev = samp.view(1, 1)
                                stega_sentence.append(int(prev))
                                x = torch.cat([x, prev], dim=1)
                                words_num += 1
                                pred_word_stem = porter.stem(pred_word.lower())
                                # print(int(samp.view(1, 1)))
                                if int(samp.view(1, 1)) == tokenizer.eos_token_id:
                                    print("break2")
                                    break
                            else:
                                print("keywords done!")
                                if alg.lower() == "ac":
                                    cur_interval, prev, num_bits_encoded = encoder(
                                        alg, prob, bit_stream, bit_index, cur_interval, **kwargs)
                                else:
                                    prev, num_bits_encoded = encoder(
                                        alg, prob, bit_stream, bit_index, **kwargs)

                                stega_sentence.append(int(prev))
                                x = torch.cat([x, prev], dim=1)
                                words_num += 1
                                stega_bit.append(
                                    bit_stream[bit_index:bit_index+num_bits_encoded])
                                bit_index += num_bits_encoded
                                bits_num += num_bits_encoded

                                if int(prev) == tokenizer.eos_token_id and bit_index == len(bit_stream_ori):
                                    print("break3")
                                    break
                        else:  # 关键词嵌入没有结束
                            # word embedding of keywords

                            guide_word_stems = [porter.stem(
                                w.lower()) for w in keywords]
                            keywords_enc = [enc_dict[w] for w in keywords]
                            number_keywords = len(keywords)

                            keywords_enc, keywords_gpt = get_keywords(
                                keywords, enc_dict, tokenizer, mode)

                            if keywords_enc and guide:
                                sim = get_sim(keywords_enc, keywords_gpt,
                                              converter_table, guarantee, mode, only_max)
                                weight = get_weight(
                                    init_weight, guarantee, t_time, c_time)
                                print(" t_time, c_time", t_time, c_time)
                                print("weight", weight)
                                k_prob = k_prob + \
                                    torch.tensor(sim*weight).cuda()
                            prob = F.softmax(k_prob, dim=-1).reshape(-1)
                            for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
                                prob[forbidden_id] = 0
                            for forbidden_id in range(256):
                                # k_prob[forbidden_id] = 0
                                prob[forbidden_id] = 0
                            prob = prob / prob.sum()

                            if bit_index > len(bit_stream_ori):  # 秘密信息也嵌入结束了
                                print("secret_bits end!")
                                # 秘密信息嵌入已经结束了 就只有关键词的嵌入
                                # probs = keywordguide()
                                samp = torch.multinomial(prob, 1)
                                # prev = torch.multinomial(prob, 1)
                                predicted_index = int(samp.view(1, 1))
                                pred_word, predicted_text, predicted_index, = get_prediction(
                                    tokenizer, stega_sentence, keywords_gpt, predicted_index, guarantee, t_time, c_time)
                                # 选择1个单词输出 不再嵌入秘密信息
                                stega_sentence.append(int(samp.view(1, 1)))
                                x = torch.cat([x, samp.view(1, 1)], dim=1)
                                words_num += 1
                                pred_word_stem = porter.stem(pred_word.lower())

                                # print("pred_word_stem", pred_word_stem)
                                guide_next = guide
                                # c_time = c_time+1
                                # t_time = t_time
                                if pred_word_stem in guide_word_stems:
                                    ind = guide_word_stems.index(
                                        pred_word_stem)
                                    print("found keyword!", keywords[ind])
                                    keywords = keywords[:ind] + \
                                        keywords[ind+1:]

                                    t_time = t_time - c_time+1
                                    c_time = 1

                            else:  # 秘密信息嵌入还没有结束

                                # keyword,probs = keywordguide()
                                # 需要通过关键词调控一些 然后输出关键词的剩余数量
                                if alg.lower() == "ac":
                                    cur_interval, prev, num_bits_encoded = encoder(
                                        alg, prob, bit_stream, bit_index, cur_interval, **kwargs)
                                else:
                                    prev, num_bits_encoded = encoder(
                                        alg, prob, bit_stream, bit_index, **kwargs)
                                '''
                                if int(prev) == tokenizer.eos_token_id:
                                    break
                                '''
                                predicted_index = int(prev)
                                pred_word, predicted_text, predicted_index, = get_prediction(
                                    tokenizer, stega_sentence, keywords_gpt, predicted_index, guarantee, t_time, c_time)

                                stega_sentence.append(predicted_index)
                                x = torch.cat([x, prev], dim=1)
                                bits_num += num_bits_encoded
                                stega_bit.append(
                                    bit_stream[bit_index:bit_index+num_bits_encoded])
                                print("stega_bit", stega_bit)
                                bit_index += num_bits_encoded
                                # pred_word = tokenizer.decode(predicted_index)
                                words_num += 1
                                pred_word_stem = porter.stem(pred_word.lower())
                                print("pred_word_stem", pred_word_stem)
                                guide_next = guide
                                # c_time = c_time+1
                                # t_time = t_time
                                if pred_word_stem in guide_word_stems:
                                    ind = guide_word_stems.index(
                                        pred_word_stem)
                                    print("found keyword!", keywords[ind])
                                    keywords = keywords[:ind] + \
                                        keywords[ind+1:]
                                    # guide_probs = guide_probs + \
                                    # [(pred_word_stem, probs[int(
                                    # prev)].item())]
                                    t_time = t_time - c_time+1
                                    c_time = 1
                            c_time = c_time + 1  # 记录当前的时间
                            # t_time = all_time-len(keywords)-tn_time
                    if tokenizer.eos_token_id in stega_sentence:
                        stega_sentence.remove(tokenizer.eos_token_id)
                    stega_text.append(tokenizer.decode(stega_sentence))
                    # 下面是一些评测指标的计算和记录结果
                    tokenized_text = stega_sentence
                    # 2_Distinct
                    counter_2 = Counter()
                    total_2 = 0
                    distinct_2 = 0
                    distinct_2, total_2, counter_2 = distinct_n(
                        tokenized_text, 2, distinct_2, total_2, counter_2)      # Need to set n
                    tmp_distinct_2 = distinct_2 / total_2

                    # 3_Distinct
                    counter_3 = Counter()
                    total_3 = 0
                    distinct_3 = 0
                    distinct_3, total_3, counter_3 = distinct_n(
                        tokenized_text, 3, distinct_3, total_3, counter_3)      # Need to set n
                    tmp_distinct_3 = distinct_3 / total_3

                    # 4_Distinct
                    counter_4 = Counter()
                    total_4 = 0
                    distinct_4 = 0
                    distinct_4, total_4, counter_4 = distinct_n(
                        tokenized_text, 4, distinct_4, total_4, counter_4)
                    # Need to set n
                    tmp_distinct_4 = distinct_4 / total_4
                    distilGPT2_perplexity = distilGPT2_perplexity_score(
                        tokenizer.decode(stega_sentence))
                    GPT2_ppl = compute_gpt2_ppl(
                        tokenizer.decode(stega_sentence))

                    print("distilGPT2_perplexity", distilGPT2_perplexity)
                    if len(keywords) != 0:
                        failure = failure+1
                    if words_num == 0:
                        words_num = 1
                    file.write({"idx": stega_idx,
                                "keywords_ori": keywords_ori,
                                "final_keywords": keywords,
                                "bit_stream": bit_stream_ori,
                                "news": prefix + prompt_text,
                                "stego": tokenizer.decode(stega_sentence),
                                "tokens": stega_sentence,
                                "bits": stega_bit,
                                "bpw": bits_num/words_num,
                                "distilGPT2_perplexity": distilGPT2_perplexity,
                                "GPT2_ppl": GPT2_ppl,
                                "distinct_2": distinct_2 / total_2,
                                "distinct_3": distinct_3 / total_3,
                                "distinct_4": distinct_4 / total_4,

                                })
                    stega_idx += 1
                    all_GPT2_ppl = all_GPT2_ppl+GPT2_ppl
                    all_distilGPT2_ppl = all_distilGPT2_ppl+distilGPT2_perplexity
                    all_bits_num = all_bits_num+bits_num
                    all_words_num = all_words_num+words_num
                    all_distinct_2 = all_distinct_2+tmp_distinct_2
                    all_distinct_3 = all_distinct_3+tmp_distinct_3
                    all_distinct_4 = all_distinct_4+tmp_distinct_4
                    # print(tokenizer.decode(stega_sentence))
                    print(stega_idx)
                file.write({"GPT2_ppl_average": all_GPT2_ppl/stega_idx,
                            "distilGPT2__ppl_average": all_distilGPT2_ppl / stega_idx,
                            "bpw_average": all_bits_num/all_words_num,
                            "distinct_2_average": all_distinct_2/stega_idx,
                            "distinct_3_average": all_distinct_3/stega_idx,
                            "distinct_4_average": all_distinct_4/stega_idx,
                            "failure_rate": failure/stega_idx,
                            })

            logger.info("bpw :{:.2f}".format(all_bits_num/all_words_num))
            # final_ppl = utils.compute_ppl("gpt2", stega_text)
            # logger.info("ppl :{:.4f}".format(all_ppl/stega_idx))
    else:
        print("no such model %s".format(args.model))


if __name__ == '__main__':

    utils.set_seed(args.seed)
    if args.mode == "G":
        generate(args, configs)
    else:
        logger.info("no such mode %s".format(args.mode))
