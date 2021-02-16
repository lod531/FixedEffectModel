import numpy as np
import pyhdfe
import pandas as pd


def demean_dataframe(df, consist_var, category_col, epsilon=1e-8, max_iter=100):
    n = df.shape[0]
    df_copy = df.copy()
    for consist in consist_var:
        mse = 10
        iter_count = 0
        demeaned_df = df.copy()
        demeans_cache = np.zeros(n, np.float64)
        while mse > epsilon:
            for cat in category_col:
                if iter_count == 0:
                    demeaned_df[consist] = df[consist] - df.groupby(cat)[consist].transform('mean')
                else:
                    demeaned_df[consist] = demeaned_df[consist] - demeaned_df.groupby(cat)[consist].transform('mean')
            iter_count += 1
            mse = np.linalg.norm(demeaned_df[consist].values - demeans_cache)
            demeans_cache = demeaned_df[consist].copy().values
            if iter_count > max_iter:
                raise RuntimeWarning('Exceeds the maximum iteration counts, please recheck dataset')
                break
        df_copy[[consist]] = demeaned_df[[consist]]
        # print(iter_count)
    return df_copy


def demean_dataframe_pyhdfe(df, consist_var, category_col, cluster_col, epsilon=1e-8, max_iter=100, drop_singletons=True):
    algo = pyhdfe.create(ids=get_np_columns(df, category_col),
                                cluster_ids=get_np_columns(df, cluster_col),
                                drop_singletons=drop_singletons,
                                degrees_method='pairwise')
    data = algo.residualize(get_np_columns(df, consist_var))
    res = pd.DataFrame.from_records(data=data, columns=consist_var)
    for name in category_col+cluster_col:
        res[name] = get_np_columns(df, [name])[~algo._singleton_indices]
    return res, algo

    
def get_np_columns(df, columns, intercept=False):
    """Helper used to retreive columns as numpy array.

    Args:
        df (pandas dataframe): dataframe containing desired columns
        columns (list of strings): list of names of desired columns.
                                    Must be a list even if only 1
                                    column is desired.
        intercept (bool): set to True if You'd like resulting numpy array
                            to have a column of 1s appended to it.
    Returns:
        2D numpy array with columns of array consisting of feature vectors,
        i.e. the first column of the result is a numpy vector of the first
        column named in columns argument.

    """
    # dataframe is a pandas datafram
    # columns is a list of column names
    # if intercept is true a column of 1s will be appended to the result matrix
    # returns columns as float64 matrix
    if columns == []: 
        return None
    else:
        res = np.expand_dims(a=df[columns[0]].to_numpy().astype('float64'), axis=1)
        if len(columns) > 1:
            for name in columns[1:]:
                res = np.c_[res, np.expand_dims(a=df[name].to_numpy().astype('float64'), axis=1)]
        if intercept:
            res = add_intercept(res)
        return res 

