from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import json
import os
import random
import heapq
import torch
import collections
import jsonlines
import csv
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Dataset(Dataset):
    def __init__(self, data, ctx_len, epoch_length_fixed, out_dir, is_uncase=True, word_level=True, min_frequency=1,
                 vocab_size=10000):
        print('building token list...', end=' ')
        if is_uncase:
            data = data.lower()
        if word_level:
            self.UNK_TOKEN = "[UNK]"
            data = " [SEP] ".join(data.split("\n"))
            data = data.strip().split()
            items = sorted(collections.Counter(data).items(),
                           key=lambda x: x[1], reverse=True)

            unique = ["[SEP]", self.UNK_TOKEN]
            remain_unique = []
            for word, freq in items:
                if word == "[SEP]":
                    continue
                if freq < min_frequency or len(unique) >= vocab_size:
                    remain_unique.append(word)
                else:
                    unique.append(word)
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(os.path.join(out_dir, 'unk_vocab.json'), "w", encoding="utf-16") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
        else:
            self.UNK_TOKEN = "\n"
            unique = sorted(list(set(data)))
        xx = 0
        xxObj = {}
        for u in unique:
            xxObj[xx] = u
            xx += 1
        with open(os.path.join(out_dir, 'vocab.json'), "w", encoding="utf-16") as vocab_file:
            vocab_file.write(json.dumps(xxObj, ensure_ascii=False))

        data_size, vocab_size = len(data), len(unique)
        print('data has %d tokens, %d unique.' % (data_size, vocab_size))
        self.stoi = {ch: i for i, ch in enumerate(unique)}
        self.itos = {i: ch for i, ch in enumerate(unique)}
        self.ctx_len = ctx_len
        self.epoch_length_fixed = epoch_length_fixed
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return self.epoch_length_fixed

    def __getitem__(self, idx):
        # cheat: pick a random spot in dataset
        i = np.random.randint(0, len(self.data) - (self.ctx_len + 1))
        chunk = self.data[i:i+self.ctx_len+1]
        dix = [self.stoi.get(s, self.stoi[self.UNK_TOKEN]) for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long,
                         device=torch.device('cuda'))
        y = torch.tensor(dix[1:], dtype=torch.long,
                         device=torch.device('cuda'))
        return x, y


class TOKENIZER():
    def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083'):
        with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
            self.word_table = json.load(result_file)

        self.vocab_size = len(self.word_table)

        self.stoi = {v: int(k) for k, v in self.word_table.items()}
        self.itos = {int(k): v for k, v in self.word_table.items()}

        self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'

        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
        # out[self.UNKNOWN_CHAR] = -float('Inf')

        lastChar = int(x[-1])

        probs = F.softmax(torch.tensor(out), dim=-1)

        if self.itos[lastChar] == '\n':
            top_p = top_p_newline
        else:
            top_p = top_p_usual

        sorted_probs, s_index = torch.sort(probs, descending=True)

        # for j in range(30):
        #     pp = sorted_probs[j].item()
        #     if pp < 0.005:
        #         break
        #     ss = self.itos[int(s_index[j])].replace('\n','_')
        #     print(f'{math.floor(pp*100):>3.0f}{ss}', end='')
        # print('')

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])

        probs[probs < cutoff] = 0
        # print("[" + str(round(cutoff,4)) + ' ' + str(round(to_float(sum(probs)),3)) + "]", end = "")

        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)

        return torch.multinomial(probs, num_samples=1)[0]


"""
thanks to :
author: Bhrigu Srivastava
website: https:bhrigu.me
"""


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        # defining comparators less_than and equals
        def __lt__(self, other):
            return self.freq < other.freq

        def __eq__(self, other):
            if (other == None):
                return False
            if (not isinstance(other, HeapNode)):
                return False
            return self.freq == other.freq

    # functions for compression:

    def make_frequency_dict(self, text):
        frequency = {}
        for character in text:
            if not character in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if (len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))
        return b

    def compress(self):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + ".bin"

        with open(self.path, 'r+') as file, open(output_path, 'wb') as output:
            text = file.read()
            text = text.rstrip()

            frequency = self.make_frequency_dict(text)
            self.make_heap(frequency)
            self.merge_nodes()
            self.make_codes()

            encoded_text = self.get_encoded_text(text)
            padded_encoded_text = self.pad_encoded_text(encoded_text)

            b = self.get_byte_array(padded_encoded_text)
            output.write(bytes(b))

        print("Compressed")
        return output_path

    """ functions for decompression: """

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-1 * extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if (current_code in self.reverse_mapping):
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

        return decoded_text

    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + "_decompressed" + ".txt"

        with open(input_path, 'rb') as file, open(output_path, 'w') as output:
            bit_string = ""

            byte = file.read(1)
            while (len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)

            decompressed_text = self.decode_text(encoded_text)

            output.write(decompressed_text)

        print("Decompressed")
        return output_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bpw(filename):
    bit_file = filename+".bit"
    text_file = filename+".txt"
    with open(bit_file, "r", encoding="utf-8") as f:
        bits_lines = f.read().split("\n")
        bits = "".join(bits_lines)
    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        words = []
        for line in lines:
            words += line.split()[1:]
    print("%s : %s" % (filename, str(len(bits)/len(words))))


def bpw_jsonlines(filename, max_num=None):
    with open(filename, "r", encoding="utf-8") as f:
        bits = []
        tokens = []
        counter = 0
        for text in jsonlines.Reader(f):
            bits += "".join(text["bits"][2:-1])
            tokens += text["tokens"][2:-1]
            counter += 1
            if max_num is not None and counter >= max_num:
                break
        print("%s : %s %s" %
              (filename, str(len(bits) / len(tokens)), str(len(bits))))


def sample_for_classification(cover_file, stego_file, out_dir, max_num=10000):
    labels = []
    texts = []
    covers = []
    stegos = []
    os.makedirs(out_dir, exist_ok=True)
    with open(cover_file, "r", encoding="utf-8") as f:
        counter = 0
        for cover in jsonlines.Reader(f):
            if counter >= max_num:
                break
            texts.append(" ".join(cover["cover"].split(" ")[1:-1]))
            covers.append(" ".join(cover["cover"].split(" ")[1:-1]))
            labels.append(0)
            counter += 1

    with open(stego_file, "r", encoding="utf-8") as f:
        counter = 0
        for stego in jsonlines.Reader(f):
            if counter >= max_num:
                break
            texts.append(" ".join(stego["stego"].split(" ")[1:-1]))
            stegos.append(" ".join(stego["stego"].split(" ")[1:-1]))
            labels.append(1)
            counter += 1

    with open(os.path.join(out_dir, "cover.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(covers))

    with open(os.path.join(out_dir, "stego.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(stegos))

    def write2file(X, Y, filename):
        datas = []
        i = 0
        with open(filename, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            for x, y in zip(X, Y):
                writer.writerow([x, y])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, train_size=0.8)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        val_texts, val_labels, test_size=0.5)
    write2file(train_texts, train_labels, os.path.join(out_dir, "train.csv"))
    write2file(val_texts, val_labels, os.path.join(out_dir, "val.csv"))
    write2file(test_texts, test_labels, os.path.join(out_dir, "test.csv"))


def sample_from_txt_and_jsonl(cover_file, stego_file, out_dir, max_num=10000, do_sample=False):
    labels = []
    texts = []
    covers = []
    stegos = []
    os.makedirs(out_dir, exist_ok=True)
    with open(cover_file, "r", encoding="utf-8") as f:
        if not do_sample:
            covers = f.read().split("\n")[:max_num]
        else:
            covers = f.read().split("\n")
            import random
            random.shuffle(covers)
            covers = covers[:max_num]
        # covers = ["".join(c.split(".")) for c in covers ]
        labels = [0] * len(covers)
        # lens = []
        # for cover in covers:
        #    lens.append(len(cover.split(" ")))
        # print(np.mean(lens), np.std(lens), np.max(lens), np.min(lens))

    with open(stego_file, "r", encoding="utf-8") as f:
        counter = 0
        for stego in jsonlines.Reader(f):
            if not do_sample:
                if counter >= max_num:
                    break
                stego_tmp = " ".join(stego["stego"].split(" ")[1:-1])
                texts.append(stego_tmp)
                for sent in stego_tmp.split("\n"):
                    if len(sent.split(" ")) == 0 or sent == "":
                        continue
                    stegos.append(sent)
                    labels.append(1)
                    counter += 1
            else:
                stego_tmp = " ".join(stego["stego"].split(" ")[1:-1])
                texts.append(stego_tmp)
                for sent in stego_tmp.split("\n"):
                    if len(sent.split(" ")) == 0 or sent == "":
                        continue
                    stegos.append(sent)
                random.shuffle(stegos)
                stegos = stegos[:max_num]
                labels += [1] * len(stegos)

    print(out_dir, len(covers), len(stegos))
    with open(os.path.join(out_dir, "cover.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(covers))

    with open(os.path.join(out_dir, "stego.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(stegos))


def compute_fine_tune_ppl(out_dir=None, sentence=None):
    model_name_or_path = out_dir
    device = "cuda"
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    nlls = []
    with torch.no_grad():
        tokens = tokenizer(sentence, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        target_ids = input_ids.clone()
        neg_log_likelihood = model(input_ids,  labels=target_ids)[0]
        nlls.append(float(neg_log_likelihood))
        ppl = np.exp(np.array(nlls).mean())
    print(f"using {model_name_or_path} calculate: ppl={ppl:.4f}")
    return ppl


model_name_or_path = "gpt2"
device = "cuda"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)


def compute_gpt2_ppl(sentence=None):
    nlls = []
    with torch.no_grad():
        tokens = tokenizer(sentence, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        target_ids = input_ids.clone()
        neg_log_likelihood = model(input_ids,  labels=target_ids)[0]
        nlls.append(float(neg_log_likelihood))
        ppl = np.exp(np.array(nlls).mean())
    print(f"using {model_name_or_path} calculate: ppl={ppl:.4f}")
    return ppl


def compute_ppl(model_name_or_path="gpt2", sentences=None):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = "cuda"
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    nlls = []
    with torch.no_grad():
        for sentence in sentences:
            tokens = tokenizer(tokenizer.bos_token +
                               sentence, return_tensors="pt")
            input_ids = tokens.input_ids.to(device)
            target_ids = input_ids.clone()
            neg_log_likelihood = model(input_ids,  labels=target_ids)[0]
            nlls.append(float(neg_log_likelihood))
    ppl = np.exp(np.array(nlls).mean())
    print(f"using {model_name_or_path} calculate: ppl={ppl:.4f}")
    return ppl


# def create_dialog_corpus(in_file_path, out_file_path):
#     with open(in_file_path, "r") as f_in, open(out_file_path, "w") as f_out:
#         datas = json.load(f_in)
#         for data in datas:
#             for i in range(len(data)):
#                 data[i] = " ".join("".join(data[i].split()))
#             f_out.write(" [turn] ".join(data) + "\n")

def sample_data(in_file_path, sample_num=1000000, do_shuffle=False):
    counter = 0
    if not do_shuffle:
        with open(in_file_path, "r") as f_in, open(in_file_path+str(sample_num), "w") as f_out:
            while True:
                sentence = f_in.readline()
                if not sentence:
                    break
                else:
                    f_out.write(sentence)
                    counter += 1
                    if counter >= sample_num:
                        break
    else:
        with open(in_file_path, "r") as f_in:
            while True:
                sentence = f_in.readline()
                if not sentence:
                    break
                else:
                    counter += 1
                    # print(counter)
            ids = np.random.choice(counter, sample_num, replace=False)
            ids = np.sort(ids)
        with open(in_file_path, "r") as f_in, open(in_file_path + str(sample_num)+"_do_shuffle", "w") as f_out:
            counter = 0
            while True:
                sentence = f_in.readline()
                if not sentence:
                    break
                else:
                    if len(ids) > 0 and counter == ids[0]:
                        # print(counter)
                        f_out.write(" ".join(sentence.strip())+"\n")
                        if len(ids) == 1:
                            break
                        else:
                            ids = ids[1:]
                    counter += 1


if __name__ == '__main__':
    compute_ppl(model_name_or_path="gpt2", sentences=[])
    # bpw_jsonlines("/data/lastness/KE-dataset/tweet/ac/stegos-encoding.jsonl")

    # sample_data("/data/lastness/LCCC-base/corpus.txt", sample_num=1000000, do_shuffle=True)
    # for dataset in ["tweet"]:
    #     cover_file = "/data/lastness/corpus/{}.txt".format(dataset)
    #     for alg in ["ac",]:
    #         stego_file = "generation/{}-gpt/{}/stegos-encoding.jsonl".format(dataset,alg)
    #         output_dir = "generation/{}-gpt/{}".format(dataset, alg)
    #         sample_from_txt_and_jsonl(cover_file, stego_file, output_dir, max_num=10000, do_sample=True)
