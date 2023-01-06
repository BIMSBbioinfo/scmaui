import subprocess
import pandas as pd
from scipy.sparse import csc_matrix
from anndata import AnnData
from scanpy import read_10x_h5
import scanpy as sc
import h5py
import numpy as np

handle = h5py.File("data/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5", "r")


def _decode(l):
    return [x.decode("ascii") for x in l]


feature_type = _decode(handle["matrix"]["features"]["feature_type"][:])
genome = _decode(handle["matrix"]["features"]["genome"][:])
ensid = _decode(handle["matrix"]["features"]["id"][:])
interval = _decode(handle["matrix"]["features"]["interval"][:])
genename = _decode(handle["matrix"]["features"]["name"][:])

pos = pd.DataFrame(
    [x.replace(":", "-").split("-") for x in interval],
    columns=["chrom", "start", "end"],
)


mat = csc_matrix(
    (
        handle["matrix"]["data"][:],
        handle["matrix"]["indices"][:],
        handle["matrix"]["indptr"][:],
    ),
    shape=handle["matrix"]["shape"][:],
).T

barcodes = pd.DataFrame(index=_decode(handle["matrix"]["barcodes"][:]))
features = pd.DataFrame(
    {
        "interval": interval,
        "genename": genename,
        "ensid": ensid,
        "genome": genome,
        "feature_type": feature_type,
        "chrom": pos.chrom,
        "start": pos.start,
        "end": pos.end,
    }
)


adata = AnnData(mat, obs=barcodes, var=features)
adata = adata[:, adata.var.chrom.apply(lambda x: x.startswith("chr"))]
adata = adata[:, ~adata.var.chrom.isin(["chrM", "chrX", "chrY"])]

gtx = adata[:, adata.var.feature_type == "Gene Expression"]
peaks = adata[:, adata.var.feature_type == "Peaks"]

gtx.write("data/gtx.h5ad")
peaks.write("data/peaks.h5ad")
