import os
from collections import OrderedDict
from anndata import AnnData
from anndata import read_h5ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import issparse, coo_matrix
from scipy.io import mmread
import tensorflow as tf

def load_batch_labels(adata, batches):
    if batches is None:
        df = pd.DataFrame({'dummybatch':['dummy']*len(barcodes)},
                           index=adata.obs.index)
    elif isinstance(batches, str) and os.path.exists(batches):
        df = pd.read_csv(batches, sep='\t', index_col=0)
    return df

def one_hot_encode_batches(adatas, batchnames):
    adata = adatas[list(adatas.keys())[0]]
    for label in batchnames:
        oh= OneHotEncoder(sparse=False).fit_transform(adata.obs[label].values.astype(str).reshape(-1,1).tolist())
        adata.obsm[label] = oh
        
    return adatas

def load_data_(data, names):
    adata = OrderedDict(),
    for datum, name in zip(data, names):
        ada = read_h5ad(datum)
        if 'view' not in ada.uns:
            ada.var.loc[:,'view'] = name.replace('/','_')
        adata[ada.var.loc[:, 'view']] = ada
    return adata

def load_data(data, outdata):
    
    adatas = {'input': OrderedDict(),
              'output': OrderedDict()}
    for datum in data:
        ada = read_h5ad(datum)
        if 'view' not in ada.uns:
            ada.uns['view'] = datum.replace('/','_')
            ada.var.loc[:, 'view'] = datum.replace('/','_')
        adatas['input'][ada.uns['view']] = ada

    if outdata is None:
        adatas['output'] = adatas['input']
    else:
        for datum in outdata:
            ada = read_h5ad(datum)
            if 'view' not in ada.uns:
                ada.uns['view'] = datum.replace('/','_')
                ada.var.loc[:, 'view'] = datum.replace('/','_')
            adatas['output'][ada.uns['view']] = ada
    return adatas


def to_sparse(x):
    if issparse(x):
        smat = x.tocoo()
    else:
        smat = coo_matrix(x)
    return smat


def to_sparse_tensor(x):
    return tf.SparseTensor(indices=np.mat([x.row, x.col]).transpose(), values=x.data, dense_shape=x.shape)


def _make_ds(x):
    ds_x = []
    for x_ in x:
        if issparse(x_):
            ds = tf.data.Dataset.from_tensor_slices(to_sparse_tensor(x_)).map(lambda x: tf.cast(tf.sparse.to_dense(x), tf.float32))
        else:
            ds = tf.data.Dataset.from_tensor_slices(tf.cast(x_, tf.float32))
        ds = ds.map(lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x))
        ds_x.append(ds)
        
    ds_x = tf.data.Dataset.zip(tuple(ds_x))
    return ds_x

def to_dataset(x, mask_x, y, mask_y, label, batch_size=64, shuffle=True):
    ds_x = _make_ds(x)
    ds_mask_x = _make_ds(mask_x)
    ds_y = _make_ds(y)
    ds_mask_y = _make_ds(mask_y)
    ds_label = _make_ds(label)

    ds_x = tf.data.Dataset.zip(((ds_x,ds_mask_x), (ds_y, ds_mask_y)))

    ds = tf.data.Dataset.zip((ds_x, ds_label))

    if shuffle:
        ds = ds.shuffle(batch_size*8)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    return ds

def to_encoder_dataset(x, mask_x, label, batch_size=64, shuffle=True):
    ds_x = _make_ds(x)
    ds_mask_x = _make_ds(mask_x)
    ds_label = _make_ds(label)

    ds_x = tf.data.Dataset.zip((ds_x,ds_mask_x))

    ds = tf.data.Dataset.zip((ds_x, ds_label))

    if shuffle:
        ds = ds.shuffle(batch_size*8)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    return ds

