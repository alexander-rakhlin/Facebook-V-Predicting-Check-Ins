# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from os.path import join, exists
from contextlib import closing
import multiprocessing as mp
import ctypes
import time
import gc

dumps = "dumps"
input_dir = "data\input"
train_file = "train_big.pkl"
test_file = "test_big.pkl"

columns_to_drop = ["place_id", "time", 'x', 'y']

box_sz = 10
th = 10
top = 10


def calculate_distance(distances):
    return distances ** -2
    

class grid(object):
    def __init__(self, n_cell_x, n_cell_y, x_p, y_p, th, top, clf):
        self.n_cell_x = n_cell_x
        self.n_cell_y = n_cell_y
        self.x_sz = box_sz / n_cell_x
        self.y_sz = box_sz / n_cell_y
        self.x_pad = self.x_sz * x_p
        self.y_pad = self.y_sz * y_p
        self.th = th
        self.top = top
        self.clf = clf
        
        
def add_time_features(df):
    """
    time related features (assuming the time = minutes)
    initial_date = "2014-01-01 01:01", arbitrary decision
    """
    initial_date = pd.Timestamp("2014-01-01 01:01")
    d_times = pd.DatetimeIndex(initial_date + pd.to_timedelta(df.time, unit="m"))
    df["hour"] = d_times.hour + d_times.minute / 60
    df["weekday"] = d_times.weekday
    df["month"] = d_times.month
    df["year"] = d_times.year

    return  df
   

def append_periodic_time(df):
    add_data = df[df["hour"] < 2.5].copy()
    add_data["hour"] += 24
    add_data2 = df[df["hour"] > 22.5].copy()
    add_data2["hour"] -= 24
    df = df.append(add_data).append(add_data2)   
    return df
    

def add_weights(df):
    df["year"] *= 10
    df["hour"] *= 4
    df["weekday"] *= 3.12
    df["month"] *= 2.12
    df["accuracy"] = np.log10(df["accuracy"]) * 10
    df["x_"] = df["x"] * 465.0 
    df["y_"] = df["y"] * 975.0    
    return df


def prepare_data(force_write=False):
    """
    Add features and dump to disk for future use
    Return pandas data frames df_train, df_test
    """
    print("Loading data ...")
    if (not force_write) & exists(join(input_dir, train_file)) &\
                         exists(join(input_dir, test_file)):
        print("Reading preprocessed files from disk.")
        df_train = pd.read_pickle(join(input_dir, train_file))
        df_test = pd.read_pickle(join(input_dir, test_file)) 
        return df_train, df_test    


    print("Creating features...")
    df_train = pd.read_csv("data/input/train.csv", dtype={"x":np.float32,
                                                          "y":np.float32, 
                                                          "accuracy":np.int16,
                                                          "time":np.int,
                                                          "place_id":np.int64},
                                                          index_col=0)

    df_test =  pd.read_csv("data/input/test.csv",  dtype={"x":np.float32, 
                                                          "y":np.float32,
                                                          "accuracy":np.int16,
                                                          "time":np.int},
                                                          index_col=0)    
    df_train = add_time_features(df_train)
    df_test = add_time_features(df_test)

    df_train = append_periodic_time(df_train)

    df_train = add_weights(df_train)
    df_test = add_weights(df_test)    
    
    print("Dumping preprocessed files to disk.")
    df_train.to_pickle(join(input_dir, train_file))
    df_test.to_pickle(join(input_dir, test_file))
    
    return df_train, df_test


def create_classifier(cl, y_sz=None):
    if cl == "xgb":
        clf = xgb.XGBClassifier(max_depth=3, learning_rate=0.1,
                                objective='multi:softmax')
    elif cl == "knn":
        assert y_sz is not None
        clf = KNeighborsClassifier(n_neighbors=np.floor((np.sqrt(y_sz)/5.3)).astype(int),
                                   weights=calculate_distance, metric='manhattan', n_jobs=-1)
    elif cl == "rf":
        clf = RandomForestClassifier(n_estimators=150, max_depth=None,
                                     min_samples_split=1, random_state=0, n_jobs=1)
    return clf                                     


def process_one_cell_df(train_cell, test_cell, g):
    """
    Return:
    ------    
    pred_labels: numpy ndarray
                 Array with the prediction of the top 3 labels for each sample
    row_ids: IDs of the samples in the submission dataframe 
    """   

    train = np.frombuffer(shared_train).reshape(train_x, train_y)
    test = np.frombuffer(shared_test).reshape(test_x, test_y)

    if (train_cell[0] >= train_cell[1]) | (test_cell[0] >= test_cell[1]):
        return None, None
    row_ids = test[test_cell[0]:test_cell[1], 0].astype(int)

    le = LabelEncoder()
    y = le.fit_transform(train[train_cell[0]:train_cell[1], 0])
    X = train[train_cell[0]:train_cell[1], 1:]

    clf = create_classifier(g.clf, y.size)
    clf.fit(X, y)
    
    X_test = test[test_cell[0]:test_cell[1], 1:]
    y_prob = clf.predict_proba(X_test)

    pred_y = np.argsort(y_prob, axis=1)[:,::-1][:,:g.top]
    pred_labels = le.inverse_transform(pred_y).astype(np.int64)
    
    labs = pd.DataFrame(pred_labels, index=row_ids)
    labs.index.name = "row_id"
    probs = pd.DataFrame(y_prob[np.arange(len(y_prob)).reshape(-1,1), pred_y], index=row_ids)
    probs.index.name = "row_id"
    
    return labs, probs


def iterate_grid_df(train_boundaries, test_boundaries, g, verbose=False):
    iteration = 0
    range_y = range(g.n_cell_y)
    range_x = range(g.n_cell_x)
    
    for y in range_y:
        for x in range_x:
            iteration += 1
            train_cell = next(train_boundaries)
            test_cell = next(test_boundaries)
            yield (g, iteration, len(range_x)*len(range_y), verbose, train_cell, test_cell)
            

def f_df(z):
    g, i, l, verbose, train_cell, test_cell = z
    labs, probs = process_one_cell_df(train_cell, test_cell, g)
    if (i % 10 == 0) & verbose:
        print("{}/{} done".format(i, l))
    return (labs, probs)


def init(shared_train_, shared_test_, train_x_, train_y_, test_x_, test_y_):
    global shared_train
    global shared_test
    global train_x
    global train_y
    global test_x
    global test_y
    shared_train = shared_train_ # must be inherited, not passed as an argument
    shared_test = shared_test_
    train_x = train_x_
    train_y = train_y_
    test_x = test_x_
    test_y = test_y_
    

def process_one_grid(train_boundaries, test_boundaries, g, verbose=False):

    it = iterate_grid_df(train_boundaries, test_boundaries, g, verbose)
    proc_num = None if g.clf == "knn" else 1
    with closing(mp.Pool(initializer=init, initargs=(shared_train, shared_test,
                                                     train_x, train_y,
                                                     test_x, test_y),
                                                     processes=proc_num)) as p:
        result = p.map(f_df, it)
        
    labs, probs = [z for z in zip(*result)]
    df_val = pd.concat(labs)
    df_prob = pd.concat(probs)
    df_val = df_val.sort_index()
    df_prob = df_prob.sort_index()
    
    return df_val, df_prob
    

def add_cell_labels(df, g, padding=True):
    t = time.time()
    
    if padding:

        l0 = np.linspace(0, box_sz, num=g.n_cell_x, endpoint=False) - g.x_pad
        l1 = l0 + g.x_sz + g.x_pad*2
        l1[-1] *= 1.01
        bins = zip(l0, l1)
        
        d = [df[(df["x"] >= l0) & (df["x"] < l1)] for l0,l1 in bins]
        xlabels = np.concatenate([i*np.ones(len(g), dtype=int) for i,g in enumerate(d)])
        d = pd.concat(d)
        d = pd.concat([d, pd.DataFrame(xlabels, index=d.index, columns=["xg"])], axis=1)

        l0 = np.linspace(0, box_sz, num=g.n_cell_y, endpoint=False) - g.y_pad
        l1 = l0 + g.y_sz + g.y_pad*2
        l1[-1] *= 1.01
        bins = zip(l0, l1)
        
        d = [d[(d["y"] >= l0) & (d["y"] < l1)] for l0,l1 in bins]
        ylabels = np.concatenate([i*np.ones(len(g), dtype=int) for i,g in enumerate(d)])
        d = pd.concat(d)
        d = pd.concat([d, pd.DataFrame(ylabels, index=d.index, columns=["yg"])], axis=1)
        
    else:
        
        bins = np.linspace(0, box_sz, g.n_cell_x+1)
        bins[-1] *= 1.01
        # float32 to avoid edge effects on bins
        xlabels = pd.cut(df["x"], bins.astype("float32"), right=False, labels=range(g.n_cell_x)).astype(int)
        xlabels.name = "xg"
        
        bins = np.linspace(0, box_sz, g.n_cell_y+1)
        bins[-1] *= 1.01
        # float32 to avoid edge effects on bins
        ylabels = pd.cut(df["y"], bins.astype("float32"), right=False, labels=range(g.n_cell_y)).astype(int)
        ylabels.name = "yg"
        
        d = pd.concat([df, xlabels, ylabels], axis=1)

    d = d.sort_values(["xg", "yg"])
    print("Adding cell labels took", int(time.time()-t), "sec")
    
    return d


def get_cell_boundaries(df):
    gr = df.groupby(["xg", "yg"])
    i = gr["accuracy"].count().cumsum().values
    idx = zip(np.hstack((0, i)), i)
    return idx

    
if __name__ == "__main__":

    mp.freeze_support()
    global shared_train
    global shared_test
    global train_x
    global train_y
    global test_x
    global test_y   


    grids = ((20, 40, "knn"), (20, 45, "knn"), (30, 45, "knn"), (40, 50, "knn"),\
             (45, 55, "xgb"), (55, 45, "xgb"), (50, 50, "xgb"),\
             (20, 50, "rf"), (30, 30, "rf"), (50, 60, "rf"))
             
    for gr in grids:
        t0 = time.time()
        n_cell_x, n_cell_y, clf = gr
        g = grid(n_cell_x, n_cell_y, 0.054, 0.06, th, top, clf)
        print("Processing grid {} ".format(gr))

        # read data        
        df_train, df_test = prepare_data()
        df_test = df_test.reset_index()
    
        cols = df_train.columns.drop(columns_to_drop).tolist() 
        train_cols = ['place_id'] + cols
        test_cols = ['row_id'] + cols

        # create shared array df_train
        df_train = add_cell_labels(df_train, g, padding=True)
        train_boundaries = get_cell_boundaries(df_train)
        
        shared_train = mp.RawArray(ctypes.c_double, df_train.shape[0]*len(train_cols))
        train = np.frombuffer(shared_train)
        train[:] = df_train.loc[:, train_cols].values.flatten()
        train_x = df_train.shape[0]
        train_y = len(train_cols)
    
        del df_train
        gc.collect() 


        # create shared array df_test
        df_test = add_cell_labels(df_test, g, padding=False)    
        test_boundaries = get_cell_boundaries(df_test)
        
        shared_test = mp.RawArray(ctypes.c_double, df_test.shape[0]*len(test_cols))
        test = np.frombuffer(shared_test)
        test[:] = df_test.loc[:, test_cols].values.flatten()
        test_x = df_test.shape[0]
        test_y = len(test_cols)
        
        del df_test
        gc.collect()    
        

        df_val, df_prob = process_one_grid(train_boundaries, test_boundaries, g, verbose=True)
        
        df_val = df_val.stack()
        df_prob = df_prob.stack()
        df = pd.concat([df_val, df_prob], axis=1).rename(columns={0:"pid", 1:"prob"})
        df.index = df.index.droplevel(1)
        df.to_pickle(join(dumps, "({}, {}){}_f".format(n_cell_x, n_cell_y, clf)))
        
        del df_val, df_prob, df
        gc.collect() 
    
        t1 = time.time()
        print("grid {} done, {} minutes elapsed".format(gr, int(t1 - t0) // 60))
