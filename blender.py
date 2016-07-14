# coding: utf-8

import numpy as np
import pandas as pd
from os.path import join, exists
import zipfile
import os
import time
import math
import multiprocessing as mp
import gc
from itertools import islice


input_dir = "dumps"
submission_path = "submission"
if not exists(input_dir): os.mkdir(input_dir)
if not exists(submission_path): os.mkdir(submission_path)
        
input_files = ['(55, 65)rf', '(50, 70)rf', '(50, 60)rf',\
               '(20, 40)knn', '(30, 45)knn', '(40, 50)knn',\
               '(45, 55)', '(50, 50)', '(55, 45)']  

chunk_sz = 10000
big_chunk_sz = chunk_sz * 400 # ~4 mio lines
aggregator = np.sum # np.sum \ np.mean (np.sum a bit better)


def iterate_data_frames(df_list, chunk_sz):
    for i,chunk in enumerate(range(0, len(df_list[0]), chunk_sz)):
        df = [d.iloc[chunk:chunk+chunk_sz, :] for d in df_list]
        df = pd.concat(df)
        df["prob"] = df["prob"].astype(float)
        yield (i, math.ceil(len(df_list[0]) / chunk_sz), df)       
    

def f_blend(x):
    i, l, df = x
    groupped = df.groupby([df.index, "pid"])
    gg = groupped["prob"].aggregate(aggregator)
    gg = gg.groupby(level=0, group_keys=False).nlargest(3)
    gg = gg.reset_index(level=1)
    s = gg.groupby(level=0).apply(lambda g: pd.Series(g["pid"].values))
    sub = s[0].map(str) + " " + s[1].map(str) + " " + s[2].map(str)
    if (i+1) % 10 == 0:
        print("{}/{} done".format(i+1, l))
        
    return sub


def iterate_big_chunks(input_files, big_chunk_sz):
    df = pd.read_pickle(join(input_dir, input_files[0]))
    l = len(df)
    del df
    for j,i in enumerate(range(0, l, big_chunk_sz)):
        df_list = []
        gc.collect()
        print("Processing big chunk {}/{}".format(j+1, math.ceil(l / big_chunk_sz)))
        for input_file in input_files:
            df = pd.read_pickle(join(input_dir, input_file)).iloc[i:i+big_chunk_sz, :].copy()
            df_list.append(df)
            gc.collect()
        del df
        gc.collect()
        assert all(df_list[0].index[::-1] + df_list[0].index == df_list[0].index[-1] + df_list[0].index[0])
        assert all([all(df_list[0].index == d.index) for d in df_list[1:]])
        yield df_list
    
    
if __name__ == '__main__':    

    h_df = pd.DataFrame()
    for input_file in input_files:
        print("Processing", input_file)
        
        df = pd.read_pickle(join(input_dir, input_file))
        df["prob"] = df["prob"].astype(float)
        assert all(df.index[::-1] + df.index == len(df)/10-1)

        r = df["prob"].values.reshape((-1, 10))
        ind = r.argsort(axis=1, kind="mergesort")[:, ::-1]
        pid = df["pid"].values.reshape((-1, 10))
        sub = pd.DataFrame(pid[np.arange(len(pid)).reshape(-1,1), ind[:, :3]],
                               index=df.index[::10])
        sub = sub[0].map(str) + " " + sub[1].map(str) + " " + sub[2].map(str)

        submission_file = join(submission_path, input_file + ".csv")
        print("Submission ready, writing", os.path.basename(submission_file), "to disk")
        sub.reset_index().to_csv(submission_file, index=False,
                                 header=["row_id", "place_id"])
        
        with zipfile.ZipFile(submission_file + '.zip', 'w') as myzip:
            myzip.write(submission_file, os.path.basename(submission_file),
                        compress_type=zipfile.ZIP_DEFLATED)
        os.remove(submission_file)


    t = time.time()
    submission = []
    for df_list in iterate_big_chunks(input_files, big_chunk_sz):  
        pool = mp.Pool()
        it = iterate_data_frames(df_list, chunk_sz)

        N = 4
        result = []
        while True:
            g2 = pool.map(f_blend, islice(it, N))
            if g2:
                result.extend(g2)
            else:
                break        

        sub = pd.concat(result)
        submission.append(sub)
        del sub
        gc.collect()
  
    submission = pd.concat(submission)
    submission_file = join(submission_path, ".".join(input_files) + "." + aggregator.__name__ + ".csv")
    print("Submission ready, writing", os.path.basename(submission_file), "to disk")
    submission.reset_index().to_csv(submission_file, index=False, 
                                    header=["row_id", "place_id"])
    
    with zipfile.ZipFile(submission_file + '.zip', 'w') as myzip:
        myzip.write(submission_file, os.path.basename(submission_file),
                    compress_type=zipfile.ZIP_DEFLATED)
    os.remove(submission_file)
    print(int(time.time() - t), "seconds")    