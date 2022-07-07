import numpy as np
import glob
import random
import torch


# Open Data
# path = 'data/'
# with open(f'{path}SBBICkm_KH.txt', 'r', encoding="utf8") as f:
#     all_words = f.readlines()

# 1. Function to Read data from txt file
def read_from_txt(path):
    with open(path, 'r', encoding="utf8") as file:
        data = file.readlines()
    return data


# Define Khmer Characters
UNI_KA = 0x1780
UNI_LAST = 0x17f9

C_START = UNI_LAST - UNI_KA + 1
C_STOP = C_START + 1
C_UNK = C_START + 2
N_CHAR = UNI_LAST - UNI_KA + 1 + 3
# N_CHAR = UNI_LAST - UNI_KA - len(UNI_BLANK) + 1 + 3


# Create functions to Generate Error Words Pairs
def str_insert(string):
    pos = random.randint(0, len(string))
    rand_c = chr(UNI_KA + random.randint(0, N_CHAR - 1 - 3))
    return string[:pos] + rand_c + string[pos:]


def str_delete(string):
    pos = random.randint(0, len(string))
    return string[:pos] + string[pos + 1:]


def str_replace(string):
    pos = random.randint(0, len(string))
    rand_c = chr(UNI_KA + random.randint(0, N_CHAR - 1 - 3))
    return string[:pos] + rand_c + string[pos + 1:]


def str_rand_err(string):
    t = random.randint(0, 3)
    if t == 0:
        return str_insert(string)
    elif t == 1:
        return str_delete(string)
    elif t == 2:
        return str_replace(string)
    else:
        return string

# Word Embedding
def str2ints(string):
    tmp = []
    for c in string:
        c = C_UNK if ord(c) < UNI_KA or ord(c) > UNI_LAST else ord(c) - UNI_KA
        tmp.append(c)
    return tmp


def onehot(ints, n_class):
    """
  ints: np (l) of int
  n_class: int
  return: np (l, n_class) of int
  """
    ints_len = len(ints)
    tmp = np.zeros((ints_len, n_class), dtype=int)
    for j, i in enumerate(ints):
        tmp[j, i] = 1
    return tmp


# Initial Tensor
def input0_tensor(b_sz):
    tmp = np.zeros((b_sz, N_CHAR), dtype=int)
    tmp[:, C_START] = 1
    return torch.tensor(tmp, dtype=torch.float32)


# Word to Tensor and Vice versa
def word2tensor(ws):
    """
    ws: list (b) of string
    return: tensor (b, max_len_str)
    """
    tmp = []
    max_len = 0
    for w in ws:
        if len(w) > max_len:
            max_len = len(w)
        tmp.append(str2ints(w))
    np_tmp = np.ones((len(ws), max_len + 1), dtype=int) * C_UNK
    for i in range(len(ws)):
        np_tmp[i, :len(tmp[i])] = np.array(tmp[i], dtype=int)
        np_tmp[i, len(tmp[i])] = C_STOP
    # print(np_tmp)
    np_tmp = np_tmp.flatten()
    np_tmp = onehot(np_tmp, N_CHAR)
    np_tmp = np_tmp.reshape((len(ws), max_len + 1, -1))
    return torch.tensor(np_tmp, dtype=torch.float32)


def label2tensor(ws):
    tmp = []
    max_len = 0
    for w in ws:
        if len(w) > max_len:
            max_len = len(w)
        tmp.append(str2ints(w))
    np_tmp = np.ones((len(ws), max_len + 1), dtype=int) * C_UNK
    for i in range(len(ws)):
        np_tmp[i, :len(tmp[i])] = np.array(tmp[i], dtype=int)
        np_tmp[i, len(tmp[i])] = C_STOP
    t_tmp = torch.tensor(np_tmp, dtype=torch.long)

    coef = np.ones((len(ws), max_len + 1))
    y_len = []
    for i in range(len(ws)):
        coef[i, len(ws[i]) + 1:] = 0
        y_len.append(len(ws[i]) + 1)

    return t_tmp, torch.tensor(coef, dtype=torch.float32), torch.tensor(y_len, dtype=torch.float32)


def tensor2str(predict):
    tmp = predict.cpu().numpy()
    lst_s = []
    for i in range(tmp.shape[0]):
        s = ''
        for c in tmp[i]:
            if c == C_STOP:
                break
            if 0 <= c <= UNI_LAST - UNI_KA:
                s += chr(UNI_KA + c)
        lst_s.append(s)
    return lst_s


# Function to set dataset size for training and testing
def p2int_selection(word_list, x_percent):
    """
    param word_list: list of words or elements to do a selection
    :param word_list:
    :param x_percent: the number of percentage that you want to select. It must be between 0 (0%) and 1 (100%).
    :return: a size of selection with the datatype of integer
    """
    return int(x_percent * len(word_list))


# Function to randomly select elements from a list
def rand_selection(items_list, number_of_selection):
    list_size = len(items_list)
    index_list = random.sample(range(list_size), number_of_selection)
    new_list = []
    for _ in range(number_of_selection):
        selected_index = index_list[_]
        selected_item = items_list[selected_index]
        # new_list.append(selected_item)
        new_list.append(selected_item.replace("\n", ""))
    return new_list


# Find Maximum word in length in a list
def find_max_len(words_list):
    word_len_list = []
    for word in words_list:
        word_len = len(word)
        word_len_list.append(word_len)

    max_len = None
    for word_len in word_len_list:
        if max_len is None or word_len > max_len:
            max_len = word_len
    return max_len

def find_files(path):
    return glob.glob(path)

def read_txt_in_folder(path):
    new_list = []
    for file in find_files(f'{path}/*.txt'):
        word_list = read_from_txt(file)
        for word in word_list:
            new_list.append(word)
    return new_list
