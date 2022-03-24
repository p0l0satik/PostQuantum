from xmlrpc.client import MAXINT
import numpy as np
import itertools
from copy import deepcopy
# from collections import *
from pyldpc.pyldpc.code import coding_matrix_systematic
def gen_H(n, r, w):
    H = np.zeros((r, n))
    for row in range(r):
        candidate = np.concatenate([np.ones((w, )), np.zeros((n-w, ))])
        # while np.sum(candidate) != w:
        #     candidate = np.random.randint(2, size=n)
        H[row] = np.random.permutation(candidate)
    print("H:\n", H)
    return H

# def gen_QCH(n, r, w):
#     H = np.zeros((r, n))


def gen_G(n, r, H):
    H_std = deepcopy(H)
    row_order = np.arange(n)
    for row in range(r-1, -1, -1):
        # print(row)
        # print("H_std before\n", H_std)
        for n_c, c in enumerate(H_std[row]):
            if c == 1:
                swp_n_c = -(r - row)
                print(r, row, n_c, swp_n_c)
                H_std[:, [n_c, swp_n_c]] = H_std[:, [swp_n_c, n_c]]
                row_order[n_c], row_order[swp_n_c] = row_order[swp_n_c], row_order[n_c]
                break
        # print("H_std\n", H_std.astype(float))
        # print(row_order)
        
        for n_r in range(r - 1, -1, -1):
            if n_r == row:
                continue
            chk_n_c = (n - r) + row
            # print("n-r:", n-r)
            # print(n_r, row, chk_n_c, H_std[n_r], H_std[n_r][chk_n_c])
            if H_std[n_r][chk_n_c] == 1:
                H_std[n_r] += H_std[row]
                H_std[n_r] %= 2
    print("H_std\n", H_std.astype(float))
    print(row_order)
    G = np.concatenate([np.eye(n-r), (H_std[:, :n-r].T %2)], axis=1)
    print("G: \n", G)
    # print(H@G.T%2, H_std@G.T%2, sep="\n")
    # print(G@H.T % 2, G@H_std.T % 2, sep="\n")

    return row_order, G, H_std

def bit_flip(n, H, y, sigma):
    synd = H_std @ y.T % 2
    # if np.all(synd == 0):
    #     return y
    ch_per_bit = np.zeros((n,))
    for n_c in range(n):
        ch_per_bit[n_c] += np.sum(synd.astype(int) & H[:, n_c].astype(int))
    # print(np.argmax(ch_per_bit))
    b = np.max(ch_per_bit) - sigma
    # print(b, ch_per_bit)
    for n_c, checks in enumerate(ch_per_bit):
        if checks >= b:
            y[n_c] += 1
            # break
    # y = np.where(ch_per_bit >= b, y, y + 1 % 2) 
    y %= 2
    # print(y)
    return y

def decode(n, H, y, sigma, n_iter = 1000):
    synd = H_std @ y.T % 2
    while sigma >= 0 and np.any(synd != 0):
        it = 0
        while it < n_iter and np.any(synd != 0):
            y = bit_flip(n, H, y, sigma)
            synd = (H_std @ y.T) % 2
            # print(synd, y)
            it += 1
        sigma -= 1

    return y

def calculate_t(n, l, G, H):
    min_d = n -  np.linalg.matrix_rank(H.T)
    print("t =", min_d)
    min_d = 1000000
    for codeword in itertools.product([0, 1], repeat=l):
        if np.all(np.array(codeword) == 0):
            continue
        # print(np.array(codeword).shape)
        d = np.sum(np.array(codeword) @ G % 2) 
        if min_d > d:
            print(d, np.array(codeword) @ G % 2)
            min_d = d
    # # print("min_d =",  min_d)
    t = (min_d - 1) // 2
    print("t = ", t)
    return t
if __name__ == "__main__":
    # n, r, w = 9602, 4801, 90 
    n, r, w = 10, 6, 2
    sigma = 0
    H = gen_H(n, r, w)
    # H = np.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 0],\
    #           [0, 1, 1, 0, 1, 1, 1, 0 ,0 ,0],\
    #           [0, 0, 0, 1, 0, 0, 0, 1, 1, 1],\
    #           [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],\
    #           [0, 0, 1, 0, 0, 1, 0, 1, 0, 1]])

    # H = np.array([[1., 0., 0., 0., 1.],[1., 1., 0., 0., 0.],[0., 0., 1., 1., 0.]])
    print("rank: ", np.linalg.matrix_rank(H))
    while np.linalg.matrix_rank(H) != r:
        H = gen_H(n, r, w)
    # row_order, G_std, H_std = gen_G(n, r, H)
    G_std, H_std = coding_matrix_systematic(H)

    t = int(calculate_t(n, n-r, G_std,  H_std))
    w = np.random.randint(2, size=n-r)
    # w = np.array([1, 0])
    c = w @ G_std % 2
    print("enc: ", c)
    err = np.concatenate([np.ones((t, )), np.zeros((n-t, ))])
    err = np.random.permutation(err)
    print("err: ", err)
    corrupted_c = (c + err) % 2
    print("cor: ", corrupted_c)
    decoded_c = decode(n, H_std, corrupted_c, sigma)
    print("enc: ", c)
    print("dec: ", decoded_c)
    print("errors: ", np.sum(c.astype(int)^decoded_c.astype(int)))