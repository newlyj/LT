import numpy as np


def lt_bpdecoder(signal, n, raw, max_iter = 1):
    # 1. get vi and cj
    # vi：the neighbor of the variable node i
    # cj: the neighbor of the check node j
    m = len(raw) 
    cji = [raw[i] for i in range(m)]
    vij = []
    for i in range(n):
        temp = []
        for j in range(m):
            if i in cji[j]:
                temp.append(j)
        vij.append(temp)  # lists corresponding LT check bits j's for LDPC bit i
    # initial
    Lch = signal
    tanh = np.tanh
    prod = np.prod
    arctan = np.arctan
    L_hij = np.zeros(shape=(n, m))
    L_fji = np.zeros(shape=(n, m))
    count = 0

    # start
    ##first Lvij
    ## I try to modify source code based on "soft decoding method for systematic raptor codes"
    while (True):
        count += 1
        for j in range(m):
            i_lists = cji[j]
            for i in i_lists:
                i_lists_copy = i_lists[:]
                i_lists_copy.remove(i)
                # LT code to LDPC code
                PI = prod(tanh(0.5 * L_hij[i_lists_copy, j]))
                L_fji[i][j] = 2 * arctan(PI * tanh(Lch[j] / 2))

        for i in range(n):
            j_lists = vij[i]  # lists corresponding LT check bits j's for LDPC bit i
            for j in j_lists:
                j_lists_copy = j_lists[:]
                j_lists_copy.remove(j)
                L_hij[i][j] = sum(L_fji[i, j_lists_copy])

        if count >= max_iter:
            break

    x = np.zeros([n])
    for i in range(n):
        j_lists = vij[i]
        x[i] = sum(L_fji[i, j_lists])
    # print(x)
    # print(np.exp(-x)/(1+np.exp(-x)))
    output = np.array(x <= 0).astype(int)
    return output
