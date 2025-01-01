import pickle
import os
from tqdm import tqdm
import numpy as np

ALIGNMENT_FILE_TMPL = "/afs/cs.stanford.edu/u/duyy/data/AudioLLM/Alignment/result/chunk_{}.pkl"
ALIGNMENT_FILE_TMPL_CTC = "/scr-ssd/duyy/Alignment/result-ctc/chunk_{}.pkl"

OUTPUT_ALIGNEMTN_FILE_TMPL_MERGED = "/scr-ssd/duyy/Alignment/result-merge/chunk_{}.pkl"

def sanity_check():
    for i in range(24):
        with open(ALIGNMENT_FILE_TMPL.format(i), 'rb') as f:
            data = pickle.load(f)
        with open(ALIGNMENT_FILE_TMPL_CTC.format(i), 'rb') as f:
            data_ctc = pickle.load(f)
        for j in tqdm(range(len(data))):
            assert np.sum(data[j]) == np.sum(data_ctc[j])
    print("OK.")

def merge():
    def remove_trailing_zeros(lst):
        while lst and lst[-1] == 0:
            lst.pop()
    os.makedirs(os.path.dirname(OUTPUT_ALIGNEMTN_FILE_TMPL_MERGED), exist_ok=True)
    for i in range(25):
        with open(ALIGNMENT_FILE_TMPL.format(i), 'rb') as f:
            data = pickle.load(f)
        with open(ALIGNMENT_FILE_TMPL_CTC.format(i), 'rb') as f:
            data_ctc = pickle.load(f)
        cnt = 0
        sum = 0
        for j in tqdm(range(len(data))):
            remove_trailing_zeros(data[j])
            # print(data[j][-1], data_ctc[j][-1])
            # import ipdb; ipdb.set_trace();
            if (len(data[j]) > 1) and (data[j][-1] > data_ctc[j][-1]):
                offset = data[j][-1] - data_ctc[j][-1]
                data[j][-1] -= offset
                data[j][-2] += offset

                cnt += 1
            sum += 1
        print(f"{cnt} / {sum} = {cnt/sum} @ {i}")
        with open(OUTPUT_ALIGNEMTN_FILE_TMPL_MERGED.format(i), 'wb') as f:
            pickle.dump(data, file=f)

if __name__ == "__main__":
    merge()