import os
import copy
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from anndata import AnnData
from anndata import read_h5ad
import anndata as ad
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import issparse, coo_matrix
from scipy.io import mmread
import scanpy as sc
import tensorflow as tf


def get_covariates(adatas, labels, dummy_labels=False):
    """Fetching covariates from anndata objects.

    Given a list of AnnData objects and a list of covariate labels,
    the function returns a list of pd.DataFrame objects holding
    the covariate information.
    For numerical data, the covariates are just used as they are.
    For categorical data, the covariates are one-hot encoded.
    For each covariate, also a dtype is returned ('numeric' or 'category').

    Parameters
    ----------
    adatas : list(anndata.AnnData)
      List of AnnData objects from which the covariates are extracted.
      Covariates should be available in at least one of the AnnData.obs annotations.
    labels : list(str)
      List of covariate names.
    dummy_labels : bool
      Indicates whether a dummy covariate should be used in place of the real covariate.
      This is used during the prediction phase, when all datapoints should be processed relative
      to a common covariate for comparability.

    Returns
    -------
    tuple(data, dtype)
      data denotes a list of pd.DataFrames (one per covariate)
      dtype identifies the covariate type as 'numeric' or 'category'
    """
    data = []
    dtype = []
    for label in labels:
        for adata in adatas:
            if label not in adata.obs.columns:
                continue
            if is_numeric_dtype(adata.obs[label]):
                data.append(adata.obs[label].values.reshape(-1, 1))
                if dummy_labels:
                    data[-1] = np.zeros_like(data[-1])
                dtype.append("numeric")
            else:
                # treat nans differently
                # categories = adata.obs[label].unique()
                traindata = [
                    [c] for c in adata.obs[label].astype(str).tolist() if not pd.isna(c)
                ]
                # print(categories)
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

                encoder.fit(traindata)
                alldata = [[c] for c in adata.obs[label].astype(str).tolist()]
                encoding = encoder.transform(alldata)
                # oh = OneHotEncoder(categories=categories,
                #                   sparse=False,
                #                   handle_unknown='ignore').fit_transform(adata.obs[label].values.astype(str).reshape(-1,1).tolist())

                data.append(encoding)
                if dummy_labels:
                    data[-1] = np.zeros_like(data[-1])
                    data[-1][:, 0] = 1
                dtype.append("category")
            break
    return data, dtype


def load_data_(data, names):
    """Helper function to load a list of anndata objects"""
    assert len(data) == len(names)
    adata = []
    for datum, name in zip(data, names):
        ada = read_h5ad(datum)

        if "view" not in ada.uns:
            ada.var.loc[:, "view"] = name.replace("/", "_")
            ada.uns["view"] = name.replace("/", "_")

        if "mask" not in ada.obsm:
            ada.obsm["mask"] = np.ones((ada.shape[0], 1))

        adata.append(ada)
    return adata


def load_data(data, names, outdata=None, outnames=None):
    """Helper function to load input and output anndata objects

    Parameters
    ----------
    data : list(str)
       List of paths pointing to h5ad input datasets.
    names : list(str)
       List of output dataset names.
    outdata : list(str) or None
       Optional list of paths pointing to h5ad output datasets.
       If not specified, output datasets are equal to input data.
    outnames : list(str) or None
       Optional list of output dataset names.

    Returns
    -------
    dict(input=list(AnnData), output=list(AnnData))
        Dictionary containing input and output datasets.
    """
    if outdata is None:
        outdata, outnames = data, names

    adatas = {"input": load_data_(data, names), "output": load_data_(outdata, outnames)}
    return adatas


def to_sparse(x):
    """Convert input array to sparse matrix."""
    if issparse(x):
        smat = x.tocoo()
    else:
        smat = coo_matrix(x)
    return smat


def to_sparse_tensor(x):
    """Convert the sparse matrix as sparse tensor"""
    return tf.sparse.reorder(
        tf.SparseTensor(
            indices=np.mat([x.row, x.col]).transpose(),
            values=x.data,
            dense_shape=x.shape,
        )
    )


def _make_ds(x):
    """Load a tensorflow dataset."""
    if issparse(x):
        ds = tf.data.Dataset.from_tensor_slices(to_sparse_tensor(x)).map(
            lambda y: tf.cast(tf.sparse.to_dense(y), tf.float32)
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices(tf.cast(x, tf.float32))
    return ds


def _make_ds_list(x):
    """Combine a list of datasets."""
    ds_x = []
    for x_ in x:
        ds = _make_ds(x_)
        ds_x.append(ds)

    ds_x = tf.data.Dataset.zip(tuple(ds_x))
    return ds_x


def _make_masked_ds(ds, mask):
    return tuple([_make_ds_list(x) for x in [ds, mask]])


def to_dataset(X, advlabels=None, condlabels=None, Y=None, batch_size=64, shuffle=True):
    """generate a tensorflow dataset"""
    x, mx = X
    ds_x = _make_masked_ds(x, mx)
    if Y is not None:
        y, my = Y
        ds_y = _make_masked_ds(y, my)
        ds_x = tf.data.Dataset.zip((ds_x, ds_y))

    ds_advlabels = ()
    if advlabels is not None and len(advlabels) > 0:
        ds_advlabel = tf.data.Dataset.zip(
            tuple([_make_ds(label) for label in advlabels or []])
        )
        ds_advlabels += (ds_advlabel,)

    ds_condlabels = ()
    if condlabels is not None and len(condlabels) > 0:
        ds_condlabel = tf.data.Dataset.zip(
            tuple([_make_ds(label) for label in condlabels or []])
        )
        ds_condlabels += (ds_condlabel,)

    ds = tf.data.Dataset.zip((ds_x, (ds_condlabels, ds_advlabels)))

    if shuffle:
        ds = ds.shuffle(batch_size * 8)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    return ds


def is_matching_indices(adata):
    refid = indices = adata["input"][0].obs.index.tolist()
    allequal = True
    for ada in adata["input"] + adata["output"]:
        if refid != ada.obs.index.tolist():
            allequal = False
            break
    return allequal


def _unionize(adata):
    """Union datasets

    Unionize all datasets w.r.t. samples that are not shared.
    That is, samples that are not shared will be filled with missing values.
    """
    if is_matching_indices(adata):
        # no action necessary
        return adata

    # obtain all indices
    indices = []
    for ada in adata["input"]:
        indices += ada.obs.index.tolist()

    indices = list(set(indices))
    df = pd.DataFrame({"idx": np.arange(len(indices))}, index=indices)
    for ada in adata["input"]:
        for col in ada.obs.columns:
            if col not in df.columns:
                df.loc[ada.obs.index.tolist(), col] = ada.obs[col]

    new_input = []
    for ada in adata["input"]:
        n = ada.shape[0]
        i = df.loc[ada.obs.index.tolist(), "idx"].values.tolist()
        X = coo_matrix((np.ones(n), (i, np.arange(n))), shape=(df.shape[0], n))
        xnew = X.dot(ada.X)
        if issparse(xnew):
            xnew = xnew.tocsr()
        new_input.append(
            AnnData(
                xnew,
                obs=df,
                var=ada.var,
                uns=ada.uns,
                obsm={"mask": X.dot(ada.obsm["mask"])},
            )
        )

    new_output = []
    for ada in adata["output"]:
        n = ada.shape[0]
        i = df.loc[ada.obs.index.tolist(), "idx"].values.tolist()
        X = coo_matrix((np.ones(n), (i, np.arange(n))), shape=(df.shape[0], n))
        xnew = X.dot(ada.X)
        if issparse(xnew):
            xnew = xnew.tocsr()
        new_output.append(
            AnnData(
                xnew,
                obs=df,
                var=ada.var,
                uns=ada.uns,
                obsm={"mask": X.dot(ada.obsm["mask"])},
            )
        )

    return {"input": new_input, "output": new_output}


def _intersect(adata):
    """intersect datasets.

    Only use common samples across all datasets
    """
    if is_matching_indices(adata):
        # no action necessary
        return adata

    # obtain all indices
    indices = adata["input"][0].obs.index.tolist()
    for ada in adata["input"] + adata["output"]:
        indices = list(set(indices).intersection(set(ada.obs.index.tolist())))

    for i, ada in enumerate(adata["input"]):
        adata["input"][i] = ada[ada.obs.index.isin(indices), :].copy()
    for i, ada in enumerate(adata["output"]):
        adata["output"][i] = ada[ada.obs.index.isin(indices), :].copy()

    return adata


class SCDataset:
    """Multi-modal dataset generator

    Parameters
    ----------
    adata : dict(input=list(AnnData), output=list(AnnData))
        Dictionary of input and output datasets in AnnData format.
    losses : list(str)
        List of losses, one per output modality.
    adversarial : list(str) or None
        List of adversarial label names. These labels should correspond
        to a sample annotation column in at least one input/output dataset.
        Default: None
    conditional : list(str) or None
        List of conditional covariate names. Covariates should correspond
        to a sample annotation column in at least one input/output dataset.
        Default: None
    union : bool
        Indicates whether non-overlapping samples/cells across multiple datasets
        should be used. If True, missing samples/cells are marked accordingly.
        If False, non-overlapping cells are removed and only overlapping cells are considered.
        Default: True.
    """

    def __init__(self, adata, losses, adversarial=None, conditional=None, union=True):
        adata = {key: [ada.copy() for ada in value] for key, value in adata.items()}
        self.union = union
        self.adata = _unionize(adata) if union else _intersect(adata)
        self.adversarial = [] if adversarial is None else adversarial
        self.conditional = [] if conditional is None else conditional
        self.losses = losses
        assert len(self.adata["output"]) == len(
            losses
        ), "Please specify a loss for each output modality."

    def __str__(self):
        s = "Inputs: non-missing/samples x features\n"
        for i, ada in enumerate(self.adata["input"]):
            x, y = ada.shape
            m = int(ada.obsm["mask"].sum())
            s += f"\t{ada.uns['view']}: "
            s += f"{m}/{x} x {y}\n"
        s += "Outputs:\n"
        for i, ada in enumerate(self.adata["output"]):
            x, y = ada.shape
            m = int(ada.obsm["mask"].sum())
            s += f"\t{ada.uns['view']}: "
            s += f"{m}/{x} x {y}\n"
        s += f"{len(self.adversarial)} Adversarials: {[x for x in self.adversarial]}\n"
        s += f"{len(self.conditional)} Conditionals: {[x for x in self.conditional]}"
        return s

    def __repr__(self):
        return str(self)

    def subset(self, cellids):
        """returns a subset of the dataset based on a list of sample/cell ids"""
        adata = {
            k: [ada[cellids, :].copy() for ada in self.adata[k]] for k in self.adata
        }
        return SCDataset(
            adata,
            self.losses,
            adversarial=self.adversarial,
            conditional=self.conditional,
            union=self.union,
        )

    def exclude(self, cellids):
        """get a subset of the dataset based on a list of sample/cell ids"""
        adata = {
            k: [ada[~ada.obs.index.isin(cellids), :].copy() for ada in self.adata[k]]
            for k in self.adata
        }
        return SCDataset(
            adata,
            self.losses,
            adversarial=self.adversarial,
            conditional=self.conditional,
            union=self.union,
        )

    def sample(self, N):
        """returns a random sample dataset containing N cells"""
        rcellids = np.random.choice(
            self.adata["input"][0].obs.index.tolist(), N, replace=False
        )
        return self.subset(rcellids)

    def modalities(self):
        """returns the modality names for the inputs and outputs"""
        inp = []
        for ada in self.adata["input"]:
            inp.append(ada.uns["view"])
        oup = []
        for ada in self.adata["output"]:
            oup.append(ada.uns["view"])
        return inp, oup

    def adversarial_config(self):
        """configuration for the adversarial labels"""
        (
            X,
            mask_x,
            Y,
            mask_y,
            advlabel,
            condlabel,
            advdtype,
            conddtype,
        ) = self._get_input_output_data(self.adata)
        shapes = {}
        shapes["adversarial_name"] = self.adversarial
        shapes["adversarial_dim"] = [x.shape[1] for x in advlabel]
        shapes["adversarial_type"] = advdtype
        return shapes

    def conditional_config(self):
        """configuration for the conditional features"""
        (
            X,
            mask_x,
            Y,
            mask_y,
            advlabel,
            condlabel,
            advdtype,
            conddtype,
        ) = self._get_input_output_data(self.adata)
        shapes = {}
        shapes["conditional_name"] = self.conditional
        shapes["conditional_dim"] = [x.shape[1] for x in condlabel]
        shapes["conditional_type"] = conddtype
        return shapes

    def shapes(self):
        """get the dataset shapes"""
        (
            X,
            mask_x,
            Y,
            mask_y,
            advlabel,
            condlabel,
            advdtype,
            conddtype,
        ) = self._get_input_output_data(self.adata)
        shapes = {}
        shapes["inputdims"] = [x.shape[1] for x in X]
        shapes["outputdims"] = [x.shape[1] for x in Y]
        return shapes

    def size(self):
        """get the dataset size"""
        _, mask_x, _, _, _, _, _, _ = self._get_input_output_data(self.adata)
        return mask_x[0].shape[0]

    def __len__(self):
        return self.size()

    def training_data(
        self, batch_size=64, validation_split=0.15, as_tf_data=True, shuffle=True
    ):
        """returns a tensorflow.data.Dataset for training"""
        # unpack the data
        X, mask_x, Y, mask_y, advlabel, condlabel, _, _ = self._get_input_output_data(
            self.adata
        )

        x_train, x_test = self._split_train_test_data(X, validation_split)
        y_train, y_test = self._split_train_test_data(Y, validation_split)
        mask_x_train, mask_x_test = self._split_train_test_data(
            mask_x, validation_split
        )
        mask_y_train, mask_y_test = self._split_train_test_data(
            mask_y, validation_split
        )
        advlabel_train, advlabel_test = self._split_train_test_data(
            advlabel, validation_split
        )
        condlabel_train, condlabel_test = self._split_train_test_data(
            condlabel, validation_split
        )

        x_train = [to_sparse(x) for x in x_train]
        x_test = [to_sparse(x) for x in x_test]
        mask_x_train = [to_sparse(x) for x in mask_x_train]
        mask_x_test = [to_sparse(x) for x in mask_x_test]
        y_train = [to_sparse(x) for x in y_train]
        y_test = [to_sparse(x) for x in y_test]
        mask_y_train = [to_sparse(x) for x in mask_y_train]
        mask_y_test = [to_sparse(x) for x in mask_y_test]

        if not as_tf_data:
            return (
                [x_train, mask_x_train],
                advlabel_train,
                condlabel_train,
                [y_train, mask_y_train],
            ), (
                [x_test, mask_x_test],
                advlabel_test,
                condlabel_test,
                [y_test, mask_y_test],
            )

        tf_X = to_dataset(
            [x_train, mask_x_train],
            advlabel_train,
            condlabel_train,
            [y_train, mask_y_train],
            shuffle=shuffle,
            batch_size=batch_size,
        )
        tf_X_test = to_dataset(
            [x_test, mask_x_test],
            advlabel_test,
            condlabel_test,
            [y_test, mask_y_test],
            shuffle=False,
        )
        return tf_X, tf_X_test

    def embedding_data(self, batch_size=64, as_tf_data=True, shuffle=False):
        """returns a tensorflow.data.Dataset for evaluating the encoder"""

        X, mask_x, _, _, alabel, condlabel, _, _ = self._get_input_output_data(
            self.adata, dummy_labels=True
        )

        if not as_tf_data:
            return [X, mask_x], alabel, condlabel

        tf_X = to_dataset(
            [X, mask_x], alabel, condlabel, shuffle=shuffle, batch_size=batch_size
        )
        return tf_X

    def imputation_data(self, batch_size=64, as_tf_data=True, shuffle=False):
        """returns a tensorflow.data.Dataset for evaluating the vae for imputation"""

        X, mask_x, Y, mask_y, alabel, condlabel, _, _ = self._get_input_output_data(
            self.adata, dummy_labels=True
        )

        if not as_tf_data:
            return [X, mask_x], alabel, condlabel, [Y, mask_y]

        tf_X = to_dataset(
            [X, mask_x],
            alabel,
            condlabel,
            [Y, mask_y],
            shuffle=shuffle,
            batch_size=batch_size,
        )
        return tf_X

    def _get_label(self, adata, col, dummy_labels=True):
        labels, dtype = get_covariates(adata, col, dummy_labels=dummy_labels)
        return labels, dtype

    def _split_train_test_data(self, x_data, validation_split):
        """Split training - validation input data"""
        x_split = [
            train_test_split(x, test_size=validation_split, random_state=42)
            for x in x_data
        ]
        x_train = [x[0] for x in x_split]
        x_test = [x[1] for x in x_split]
        return x_train, x_test

    def _split_train_test(self, x_data, mask, label, validation_split):
        """Split training - validation (data + labels)"""
        x_train, x_test = self._split_train_test_data(x_data, validation_split)
        mask_train, mask_test = self._split_train_test_data(mask, validation_split)
        label_train, label_test = self._split_train_test_data(label, validation_split)
        return x_train, x_test, mask_train, mask_test, label_train, label_test

    def _get_data(self, adata):
        x_data = [to_sparse(ada.X) for ada in adata]
        mask = []
        for ada in adata:
            m = to_sparse(ada.obsm["mask"])
            mask.append(m)
        return x_data, mask

    def _get_input_data(self, adata, dummy_labels=False):
        X, mask_x, _, _, _, condlabel, _, _ = self._get_input_output_data(
            adata, dummy_labels=False
        )
        return x_data, mask_x, condlabel

    def _get_input_output_data(self, adata, dummy_labels=False):
        x_data, mask_x = self._get_data(adata["input"])
        y_data, mask_y = self._get_data(adata["output"])
        advlabels, advdtype = self._get_label(
            adata["input"], self.adversarial, dummy_labels=dummy_labels
        )
        condlabels, conddtype = self._get_label(
            adata["input"], self.conditional, dummy_labels=dummy_labels
        )
        return (
            x_data,
            mask_x,
            y_data,
            mask_y,
            advlabels,
            condlabels,
            advdtype,
            conddtype,
        )


class SCMauiDataset(SCDataset):
    pass


def combine_modalities(adatas):
    """Combine a list of AnnData objects to a single AnnData object"""
    iadata = ad.concat([ada for ada in adatas], axis=1)
    for ada in adatas:
        for col in ada.obs.columns:
            if col not in iadata.obs.columns:
                iadata.obs.loc[:, col] = ada.obs[col]

    new_view = [ada.uns["view"] for ada in adatas]
    iadata.uns["view"] = "-".join(new_view)
    return iadata


def split_modalities(adata, split_by):
    """Split a dataset into multiple datasets based on a cell annotation column."""
    adatas = []
    names = adata.var.loc[:, split_by].unique()
    for name in names:
        adatas.append(adata[:, adata.var[split_by] == name].copy())
    return adatas


if __name__ == "__main__":
    adafile = [
        "/local/wkopp/scmaui_mskcc/10xgenomics/data/gtx.h5ad",
        "/local/wkopp/scmaui_mskcc/10xgenomics/data/peaks.h5ad",
    ]
    adatas = load_data(adafile, ["gtx", "peaks"])

    dataset = SCDataset(
        adatas, adversarial=["logreads", "sample"], conditional=["logreads", "sample"]
    )
    print(dataset)

    dataset = SCDataset(adatas)
    print(dataset)

    adatas["input"][0] = adatas["input"][0][:10, :].copy()
    dataset = SCDataset(adatas, union=False)
    print(dataset)

    adatas = load_data(adafile, ["gtx", "peaks"])
    adatas["input"][0] = adatas["input"][0][:10, :].copy()
    dataset = SCDataset(adatas, union=True)
    print(dataset)

    adatas = load_data(adafile, ["gtx", "peaks"])
    adatas["input"][0] = AnnData(
        adatas["input"][0].X,
        var=adatas["input"][0].var,
        obsm={"mask": np.ones((adatas["input"][0].shape[0], 1))},
        uns=adatas["input"][0].uns,
    )
    dataset = SCDataset(adatas, union=True)
    print(dataset)
