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


def parse_arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Train or Generate",
                        choices=["T", "G", "E"], default="G")
    parser.add_argument("--config_path", type=str,
                        default="generate_configs/Yahoo-gpt-generate_wo_prompt_hc.ini")
    # Yahoo-gpt-generate_wo_prompt.ini
    # Yahoo-gpt-generate.ini
    # gpt-generate.ini
    parser.add_argument("--gpuid", type=str, default="0")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--out_dir", type=str,
                        default="/data/lastness/out-1122")
    return parser.parse_args()


args = parse_arg_main()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
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
    embedding = configs.get("generate", "embedding")
    converter_table = np.load(str(os.path.dirname(
        os.path.abspath(__file__))) + '/data/converter_table_' + str(embedding) + '.npy')
    # print(trainfile)
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

        with open(configs.get("generate", "bit_filepath"), 'r', encoding='utf8') as f:
            bit_stream_ori = f.read().strip()
        bit_stream = list(bit_stream_ori)
        bit_stream = ''.join(bit_stream)
        bit_index = int(torch.randint(0, high=10000, size=(1,)))
        max_length = configs.getint("generate", "max_length")
        generate_num = configs.getint("generate", "generate_num")
        alg = configs.get("generate", "alg")
        reference_filepath = configs.get("generate", "reference_filepath")
        kwargs = encoder_configs(configs, alg)

        outfile = alg+"_"+generatefile + "_"
        prompt = configs.get("gpt2", "prompt")
        model.eval()

        with torch.no_grad():
            stega_text = []
            stega_idx = 0
            with jsonlines.open(os.path.join("generate_result_wo_prompt", outfile + ".jsonl"), "w") as file:
                stega_idx = 0
                all_words_num = 0
                all_bits_num = 0
                all_GPT2_ppl = 0
                all_distilGPT2_ppl = 0
                all_distinct_2 = 0
                all_distinct_3 = 0
                all_distinct_4 = 0

                # Create file containing the keyword embeddings
                save_path_dict = os.path.join(
                    "data", 'dict_wo_unk' + reference_filepath.split("/")[-1].split(".")[-2] + embedding + '.pkl')
                if not os.path.isfile(save_path_dict):
                    create_enc_dict_jsonl_wo_unk(
                        reference_filepath, embedding)
                with open(save_path_dict, 'rb') as f:
                    enc_dict = pickle.load(f)
                secret_sets = get_keywordsets_bitstream_jsonl_wo_unk(
                    file_name=reference_filepath, enc_dict=enc_dict)
                # while len(stega_text) < generate_num:
                for j, keyword_set in enumerate(secret_sets):
                    stega_bits = []
                    words_num = 0
                    bits_num = 0
                    in_text, keywords, bit_stream_ori = keyword_set
                    if len(bit_stream) <= max_length * math.log2(tokenizer.vocab_size):
                        bit_stream_shuffle = list(bit_stream_ori)
                        random.shuffle(bit_stream_shuffle)
                        bit_stream += "".join(bit_stream_shuffle)
                    # sample start word
                    stega_sentence = []
                    # TODO begin

                    prefix = in_text
                    prompt_text = prompt
                    if len(in_text.split()) > 512:
                        prefix = (" ").join(in_text.split()[0:64])
                    print(len(prefix.split()))
                    encoded_prompt = tokenizer.encode(tokenizer.bos_token + prefix + prompt_text,
                                                      add_special_tokens=False,
                                                      return_tensors="pt")
                    encoded_prompt = encoded_prompt.to(device)
                    input_ids = encoded_prompt
                    print(prefix)
                    stega_bit = []
                    logits = model(input_ids).logits[:, -1, :]
                    logits -= logits.max()
                    probs = torch.exp(logits)
                    for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
                        probs[:, forbidden_id] = 0
                    for forbidden_id in range(256):
                        probs[:, forbidden_id] = 0
                    samp = torch.multinomial(probs, 1)
                    stega_sentence.append(int(samp.view(1, 1)))
                    x = torch.cat([input_ids, samp], dim=1)
                    if alg.lower() == "ac":
                        max_val = 2 ** kwargs["precision"]  # num of intervals
                        # max_val = 2**52
                        cur_interval = [0, max_val]

                    for i in range(max_length - 1):

                        if '_EOS' in stega_sentence:
                            break

                        # conditional probability distribution
                        # todo begin
                        log_prob = model(x).logits[:, -1, :]
                        log_prob -= log_prob.max()
                        prob = torch.exp(log_prob).reshape(-1)
                        for forbidden_id in range(256):
                            prob[forbidden_id] = 0
                        # todo end
                        prob = prob / prob.sum()
                        # early stop generation
                        if alg.lower() == "ac":
                            cur_interval, prev, num_bits_encoded = encoder(
                                alg, prob, bit_stream, bit_index, cur_interval, **kwargs)
                        else:
                            prev, num_bits_encoded = encoder(
                                alg, prob, bit_stream, bit_index, **kwargs)
                        if int(prev) == tokenizer.eos_token_id:
                            break
                        stega_sentence.append(int(prev))
                        x = torch.cat([x, prev], dim=1)
                        bit_index += num_bits_encoded
                        words_num += 1
                        bits_num += num_bits_encoded
                        stega_bit.append(
                            bit_stream[bit_index:bit_index+num_bits_encoded])

                    if tokenizer.eos_token_id in stega_sentence:
                        stega_sentence.remove(tokenizer.eos_token_id)

                    stega_text.append(tokenizer.decode(stega_sentence))
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

                    if distinct_2 == 0 and distinct_3 == 0 and distinct_4 == 0:
                        print(all_GPT2_ppl)
                        print(all_distilGPT2_ppl)
                        distilGPT2_perplexity = 0
                        GPT2_ppl = 0
                        print("all_GPT2_ppl", all_GPT2_ppl)
                        print("all_distilGPT2_ppl", all_distilGPT2_ppl)
                    else:
                        distilGPT2_perplexity = distilGPT2_perplexity_score(
                            tokenizer.decode(stega_sentence))
                        GPT2_ppl = compute_gpt2_ppl(
                            tokenizer.decode(stega_sentence))

                    print("distilGPT2_perplexity", distilGPT2_perplexity)

                    if words_num == 0:
                        words_num = 1
                    file.write({"idx": stega_idx,
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
