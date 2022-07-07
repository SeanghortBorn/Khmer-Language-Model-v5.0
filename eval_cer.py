import numpy as np

def edit_dist(s, t):
    table = np.zeros((len(s)+1, len(t)+1), dtype=np.int32)
    for i in range(len(s)+1):
        table[i, 0] = i
    for j in range(len(t)+1):
        table[0, j] = j
    for i in range(1, len(s)+1):
        for j in range(1, len(t)+1):
            if s[i-1]==t[j-1]:
                sub_cost = 0
            else:
                sub_cost = 1
            table[i, j] = min(table[i-1, j]+1, #deletion
                              table[i, j-1]+1, #insertion
                              table[i-1, j-1]+sub_cost) #substitution
    return table[len(s), len(t)]

def compute_cer(t1, t2):
    """
    :param t1: list (b) of string (predicted)
    :param t2: list (b) of string (gt)
    :return: float [0, 1]
    """
    sum_err = 0
    sum_len = 0
    for s1, s2 in zip(t1, t2):
        sum_err += edit_dist(s1, s2)
        sum_len += len(s2)
    return sum_err*1.0/sum_len

if __name__=='__main__':
    path = 'data-2/'
    with open(path+'output.txt', 'r', encoding='utf-8') as f:
        t1 = f.readlines()
        t1 = [_[:-1] for _ in t1]
    with open(path+'gt.txt', 'r', encoding='utf-8') as f:
        t2 = f.readlines()
        t2 = [_[:-1] for _ in t2]

    cer = compute_cer(t1, t2)
    print('CER: %.2f' % (cer*100))