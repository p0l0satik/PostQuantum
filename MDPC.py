from xmlrpc.client import MAXINT
import numpy as np
import itertools
from copy import deepcopy
from  pyldpc import coding_matrix_systematic, binaryrank

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys 

def gen_H(n, r, w):
    H = np.zeros((r, n))
    for row in range(r):
        candidate = np.concatenate([np.ones((w, )), np.zeros((n-w, ))])
        H[row] = np.random.permutation(candidate)
    return H

def gen_G(n, r, H):
    H_std = deepcopy(H)
    row_order = np.arange(n)
    #compute an Identity matix in the back of H
    for row in range(r-1, -1, -1):
        #find a column with a one at a specific row and swap it
        for n_c, c in enumerate(H_std[row]):
            if c == 1:
                swp_n_c = -(r - row)
                H_std[:, [n_c, swp_n_c]] = H_std[:, [swp_n_c, n_c]]
                row_order[n_c], row_order[swp_n_c] = row_order[swp_n_c], row_order[n_c]
                break
        #clear the ones up and below
        for n_r in range(r - 1, -1, -1):
            if n_r == row:
                continue
            chk_n_c = (n - r) + row
            if H_std[n_r][chk_n_c] == 1:
                H_std[n_r] += H_std[row]
                H_std[n_r] %= 2
    G = np.concatenate([np.eye(n-r), (H_std[:, :n-r].T %2)], axis=1)
    return G, H_std

def bit_flip(n, H, y, sigma):
    synd = H @ y.T % 2
    ch_per_bit = np.zeros((n,))
    #calulate wrong checks per bit
    for n_c in range(n):
        ch_per_bit[n_c] += np.sum(synd.astype(int) & H[:, n_c].astype(int))
    b = np.max(ch_per_bit) - sigma
    for n_c, checks in enumerate(ch_per_bit):
        if checks >= b:
            y[n_c] += 1
            break
    y %= 2
    return y

def decode(n, H, y, sigma=0, n_iter = 150):
    synd = H @ y.T % 2
    while sigma >= 0 and np.any(synd != 0):
        it = 0
        while it < n_iter and np.any(synd != 0):
            y = bit_flip(n, H, y, sigma)
            synd = (H @ y.T) % 2
            it += 1
        sigma -= 1
    return y

def simulate_t(n, r, w, n_codes = 10, n_words = 10):
    error_rate = []
    for t in range(0, n // 4 + 1):
        error_rate_t = []

        for code in range(n_codes):
            H = gen_H(n, r, w)
            G_std, H_std = gen_G(n, r, H)

            for word in range(n_words):

                in_word = np.random.randint(2, size=n-r)
                c = in_word @ G_std % 2
                err = np.concatenate([np.ones((t, )), np.zeros((n-t, ))])
                err = np.random.permutation(err)
                corrupted_c = (c + err) % 2
                decoded_c = decode(n, H_std, corrupted_c)
                dec_err = np.sum(c.astype(int)^decoded_c.astype(int))
                error_rate_t.append(dec_err / len(c))
        error_rate.append(np.mean(np.array(error_rate_t)))
    print(error_rate)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(error_rate))
    ax.set_title("MPDC-({}, {}, {})".format(n, r, w))
    ax.set_xlabel('t')
    ax.set_ylabel('BER')

    plt.savefig("./{}_{}_{}_code.png".format(n, r, w))
if __name__ == "__main__":
    n, r, w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t = simulate_t(n, r, w)