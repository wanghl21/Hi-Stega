import torch
from utils import HuffmanCoding
import numpy as np
from configparser import ConfigParser


# MSB
# e.g. [0, 1, 1, 1] looks like 0111=7
def msb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits[::-1]):
        res += bit * (2 ** i)
    return res


def msb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in strlist]

# lsb
# e.g. [0, 1, 1, 1] looks like 1110=14


def lsb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def lsb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


def near(alist, anum):
    up = len(alist) - 1
    print("up", up)
    if up == 0:
        return 0

    bottom = 0
    while up - bottom > 1:
        index = int((up + bottom) / 2)
        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index
    if up - bottom == 1:
        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up

    return index


def find_nearest_list(prob, delta,):
    diff = (np.array(prob) - delta)
    tmp_idx = np.argmin(diff**2)
    if prob[tmp_idx] < delta:
        return_list = [tmp_idx]
        if tmp_idx == len(prob) - 1:
            pass
        else:
            tmp_sum = prob[tmp_idx]
            for i in range(tmp_idx+1, len(prob)-1):
                if delta > (tmp_sum + prob[i]):
                    tmp_sum += prob[i]
                    return_list.append(i)
        return return_list
    elif tmp_idx >= len(prob)-2:
        return [tmp_idx]
    else:
        new_idx = tmp_idx + 1
        idx = [new_idx]
        idx += find_nearest_list(prob[new_idx+1:], delta-prob[new_idx])
        for i in range(1, len(idx)):
            idx[i] += new_idx+1
        # return idx
        if (delta-np.sum(np.array(prob)[idx]))**2 > diff[tmp_idx]**2:
            return [tmp_idx]
        else:
            return idx


def find_nearest(prob, delta,):
    diff = (np.array(prob) - delta)
    tmp_idx = np.argmin(diff**2)
    if prob[tmp_idx] < delta:
        return tmp_idx
    elif tmp_idx >= len(prob)-2:
        return tmp_idx
    else:
        new_idx = tmp_idx+1
        idx = find_nearest(prob[new_idx+1:], delta-prob[new_idx])
        idx += new_idx+1
        return idx


def grouping(prob):
    # prob = prob/prob.sum()
    prob, indices = prob.sort(descending=True)
    prob = prob.tolist()
    indices = indices.tolist()
    mean = 0.5
    # initialize
    groups = [[0, []], [0, []]]
    if prob[0] > mean:
        groups[0] = [prob[0], [indices[0]]]
        groups[1] = [1-prob[0], indices[1:]]
    else:
        groups[0] = [prob[0], [indices[0]]]
        del prob[0]
        del indices[0]
        delta = mean - groups[0][0]
        # while delta >= error_threshold and abs(delta-prob[-1]) > 1/2 * delta:
        # while prob[-1] < 2 * delta:
        #     idx = find_nearest(prob, delta)
        #     groups[0][0] += prob[idx]
        #     groups[0][1].append(indices[idx])
        #     del prob[idx]
        #     del indices[idx]
        while prob[-1] < 2*delta:
            idx_list = find_nearest_list(prob, delta)
            sorted_idx_list = np.sort(idx_list)[::-1]
            for idx in sorted_idx_list:
                groups[0][0] += prob[idx]
                groups[0][1].append(indices[idx])
                del prob[idx]
                del indices[idx]
            delta = mean - groups[0][0]
            # delta = abs(mean - groups[0][0])
        groups[1][0] = 1 - groups[0][0]
        groups[1][1] = indices
    return groups


def FLC_encoder(prob, bit_stream, bit_index, bit, **kwargs):
    prob, indices = prob.sort(descending=True)
    prob = prob[:2 ** bit]
    indices = indices[:2 ** bit]
    bits_list = list(bit_stream[bit_index:bit_index+bit])
    prev = indices[msb_bits2int([int(b) for b in bits_list])]
    return prev, bit


def AC_encoder(prob, bit_stream, bit_index, cur_interval, precision=52, **kwargs):
    prob, indices = prob.sort(descending=True)
    # prob = prob[:2 ** Generation_Configs.bit]
    # indices = indices[:2 ** Generation_Configs.bit]
    # arithmetic coding
    cur_int_range = cur_interval[1] - cur_interval[0]  # 区间的大小  2^26
    cur_threshold = 1 / cur_int_range  # 每个区间多大
    if prob[-1] < cur_threshold:
        k = max(2, (prob < cur_threshold).nonzero()[0].item())
        prob = prob[:k]
        indices = indices[:k]

    prob = prob / prob.sum()  # 截断后线性归一化
    prob = prob.double()
    prob *= cur_int_range  # 概率转换为多少个区间
    prob = prob.round().long()  # 四舍五入取整，区间数描述的概率

    cum_probs = prob.cumsum(0)  # 前面所有项的和的序列区间数描述的分布函数，按理讲最后应该与区间数相同
    overfill_index = (cum_probs > cur_int_range).nonzero()
    if len(overfill_index) > 0:
        cum_probs = cum_probs[:overfill_index[0]]  # 去掉最后一个概率
    cum_probs += cur_int_range - cum_probs[-1]  # 分布函数加到和区间数相等，区间数表示的分布函数

    cum_probs += cur_interval[0]  # 分布函数的第一项从左区间开始

    # 取了52位，但不是编码这52位，是用这52位锁定一个位置
    message_bits = bit_stream[bit_index: bit_index + precision]
    message_bits = [int(_) for _ in message_bits]
    message_idx = msb_bits2int(message_bits)
    selection = (cum_probs > message_idx).nonzero()[0].item()
    # message_idx_left = bits2int(reversed(message_bits))  # reverse只是为了计算int
    # message_idx_right = message_idx_left+1
    # selection_left_bound = (cum_probs > message_idx_left).nonzero()[0].item()  # 选择的单词的索引，int，选择第几个单词
    # if cum_probs[-1] > message_idx_right:
    #     selection_right_bound = (cum_probs > message_idx_right).nonzero()[0].item()
    # else:
    #     selection_right_bound = (cum_probs >= message_idx_right).nonzero()[0].item()
    # if selection_right_bound == selection_left_bound:
    #     selection_right_bound = selection_left_bound + 1
    # selection = torch.multinomial(prob[selection_left_bound:selection_right_bound]/prob[selection_left_bound:selection_right_bound].sum(), 1) \
    #                 + selection_left_bound

    new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[
        0]  # 新的左区间 如果选了第一个单词（selection=0）就代表不需要动区间的左边界
    new_int_top = cum_probs[selection]

    new_int_bottom_bits_inc = list(msb_int2bits(
        new_int_bottom, precision))  # 二进制的下边界
    new_int_top_bits_inc = list(msb_int2bits(
        new_int_top - 1, precision))  # 二进制的上边界

    num_bits_encoded = num_same_from_beg(
        new_int_bottom_bits_inc, new_int_top_bits_inc)

    # 新二进制区间
    new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [
        0] * num_bits_encoded
    new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [
        1] * num_bits_encoded

    cur_interval[0] = msb_bits2int(new_int_bottom_bits)  # 新的区间
    # +1 here because upper bound is exclusive
    cur_interval[1] = msb_bits2int(new_int_top_bits) + 1
    prev = indices[selection].view(1, 1)  # 一个数，代表选了哪个单词
    return cur_interval, prev, num_bits_encoded


def HC_encoder(prob, bit_stream, bit_index, bit, **kwargs):
    prob, indices = prob.sort(descending=True)
    prob = prob[:2 ** bit]
    indices = indices[:2 ** bit]

    prob_dict = {i: float(p) for i, p in enumerate(prob)}
    hf = HuffmanCoding()
    hf.make_heap(prob_dict)
    hf.merge_nodes()
    hf.make_codes()
    for hf_code in hf.reverse_mapping.keys():
        if hf_code == bit_stream[bit_index:bit_index + len(hf_code)]:
            num_bits_encoded = len(hf_code)
            prev = indices[hf.reverse_mapping[hf_code]].view(1, 1)
            return prev, num_bits_encoded


def ADG_encoder(prob, bit_stream, bit_index, **kwargs):
    device = prob.device
    prob, indices = prob.sort(descending=True)
    # start recursion
    bit_tmp = 0
    while prob[0] <= 0.5:
        # embedding bit
        bit = 1
        while (1 / 2 ** (bit + 1)) > prob[0]:
            bit += 1
        mean = 1 / 2 ** bit
        # dp
        prob = prob.tolist()
        indices = indices.tolist()
        result = []
        for i in range(2 ** bit):
            result.append([[], []])
        for i in range(2 ** bit - 1):
            result[i][0].append(prob[0])
            result[i][1].append(indices[0])
            del (prob[0])
            del (indices[0])
            while sum(result[i][0]) < mean:
                delta = mean - sum(result[i][0])
                index = near(prob, delta)
                if prob[index] - delta < delta:
                    result[i][0].append(prob[index])
                    result[i][1].append(indices[index])
                    del (prob[index])
                    del (indices[index])
                else:
                    break
            mean = sum(prob) / (2 ** bit - i - 1)
        result[2 ** bit - 1][0].extend(prob)
        result[2 ** bit - 1][1].extend(indices)
        # read secret message
        bit_embed = [int(_) for _ in bit_stream[bit_index +
                                                bit_tmp:bit_index + bit_tmp + bit]]
        int_embed = msb_bits2int(bit_embed)
        # updating
        prob = torch.FloatTensor(result[int_embed][0]).to(device)
        indices = torch.LongTensor(result[int_embed][1]).to(device)
        prob = prob / prob.sum()
        prob, _ = prob.sort(descending=True)
        indices = indices[_]
        bit_tmp += bit

    prev = indices[int(torch.multinomial(prob, 1))].view(1, 1)
    num_bits_encoded = bit_tmp
    return prev, num_bits_encoded


def ADG_V2_encoder(prob, bit_stream, bit_index, epsilon, max_bit, **kwargs):
    mean = 0.5
    # ori_prob = prob
    prob, indices = prob.sort(descending=True)
    # prob_sum = prob.sum()
    acc_prob_sum = 1
    # start recursion
    bit_tmp = 0
    epsilon = epsilon
    groups = grouping(prob)
    # indices test
    # if abs(groups[0][0]*acc_prob_sum - ori_prob[indices[groups[0][1]]].sum()) > 1e-6:
    #     print()
    while (abs(groups[0][0] - mean) <= epsilon * (2**bit_tmp)) and abs(groups[0][0] - mean) < mean and bit_tmp < max_bit:
        # select groups
        bit = bit_stream[bit_index+bit_tmp]
        if bit_stream[bit_index+bit_tmp] == '0':
            prob = prob[groups[0][1]]
            indices = indices[groups[0][1]]
        else:
            prob = prob[groups[1][1]]
            indices = indices[groups[1][1]]
        prob_sum = prob.sum()
        acc_prob_sum *= prob_sum
        bit_tmp += 1
        prob = prob/prob_sum
        groups = grouping(prob)
        # indices test
        # if abs(groups[0][0]*acc_prob_sum - ori_prob[indices[groups[0][1]]].sum()) > 1e-12:
        #     print()
    prev = indices[int(torch.multinomial(prob, 1))].view(1, 1)
    num_bits_encoded = bit_tmp
    return prev, num_bits_encoded


def PLAIN_encoder(prob, topk=50000, topp=0.99, do_sample=True, **kwargs):
    topk = int(topk)
    topp = float(topp)
    do_sample = True if isinstance(
        do_sample, bool) or do_sample.lower() == "true" else False
    # topk = Generation_Configs.get("topk", 1)
    # topp = Generation_Configs.get("topp", 1)
    # do_sample = Generation_Configs.get("do_sample", True)
    prob, indices = prob.sort(descending=True)
    prob = prob/prob.sum()
    cum_prob = prob.cumsum(0)
    prob_ = prob[cum_prob <= topp]
    prob = prob_[:topk] if len(prob_) > 0 else prob[:1]
    if do_sample:
        prev = indices[int(torch.multinomial(prob, 1))].view(1, 1)
    else:
        prob = torch.ones_like(prob)
        prev = indices[int(torch.multinomial(prob, 1))].view(1, 1)
    return prev, 0


def encoder(alg, prob, bit_stream, bit_index, cur_interval=None, **kwargs):
    if alg.lower() == "plain":
        return PLAIN_encoder(prob, **kwargs)
    if alg.lower() == "flc":
        return FLC_encoder(prob, bit_stream, bit_index, **kwargs)
    if alg.lower() == "hc":
        return HC_encoder(prob, bit_stream, bit_index, **kwargs)
    if alg.lower() == "ac":
        return AC_encoder(prob, bit_stream, bit_index, cur_interval, **kwargs)
    if alg.lower() == "adg":
        return ADG_encoder(prob, bit_stream, bit_index, **kwargs)
    if alg.lower() == "adgv2":
        return ADG_V2_encoder(prob, bit_stream, bit_index, **kwargs)
    raise ValueError("no such algorithm")


def encoder_configs(configs: ConfigParser, alg):
    kwargs = {}
    for k in configs.options("generate"):
        if k not in ["bit_filepath", "model_name", "max_length", "generate_num", "alg"]:
            kwargs[k] = configs.get("generate", k, )
    if alg.lower() == "plain":
        new_kwargs = {}
        for k in ["topp", ]:
            if kwargs.get("topp") is not None:
                new_kwargs[k] = configs.getfloat("generate", "topp")
        for k in ["topk", ]:
            if kwargs.get("topk") is not None:
                new_kwargs[k] = configs.getint("generate", "topk")
        for k in ["do_sample", ]:
            if kwargs.get("do_sample") is not None:
                new_kwargs[k] = configs.getboolean("generate", "do_sample")
        return new_kwargs
    if alg.lower() == "flc":
        new_kwargs = {}
        for k in ["bit", ]:
            if kwargs.get("bit") is not None:
                new_kwargs[k] = configs.getint("generate", "bit")
        return new_kwargs
    if alg.lower() == "hc":
        new_kwargs = {}
        for k in ["bit", ]:
            if kwargs.get("bit") is not None:
                new_kwargs[k] = configs.getint("generate", "bit")
        return new_kwargs
    if alg.lower() == "ac":
        new_kwargs = {}
        for k in ["precision", ]:
            if kwargs.get("precision") is not None:
                new_kwargs[k] = configs.getint("generate", "precision")
        return new_kwargs
    if alg.lower() == "adg":
        return {}
    if alg.lower() == "adgv2":
        new_kwargs = {}
        for k in ["epsilon", ]:
            if kwargs.get("epsilon") is not None:
                new_kwargs[k] = configs.getfloat("generate", "epsilon")
        for k in ["max_bit", ]:
            if kwargs.get("max_bit") is not None:
                new_kwargs[k] = configs.getint("generate", "max_bit")
        return new_kwargs
    raise ValueError("no such algorithm")
