import numpy as np
import scanpy as sc
from anndata import concat

# get the rna-seq measurements of the multi-modal measurements
# and append scrna-seq from single-modal mearurements
mgtx = sc.read_h5ad("data/gtx.h5ad")
sgtx = sc.read_10x_h5("data/scrna_5k_pbmc_v3_filtered_feature_bc_matrix.h5")
mgtx.obs.loc[:, "sample"] = "rna+atac"
sgtx.obs.loc[:, "sample"] = "rna only"

# unify gene features
mgtx.var = mgtx.var.reset_index().set_index("ensid")
# mgeneact.var = mgeneact.var.set_index('ensid')
sgtx.var = sgtx.var.reset_index().set_index("gene_ids")

s1 = set(mgtx.var.index.tolist())
s2 = set(sgtx.var.index.tolist())
genes = list(s1.intersection(s2))

mgtx = mgtx[:, genes].copy()
sgtx = sgtx[:, genes].copy()
all_gtx = concat([sgtx, mgtx], axis=0)
all_gtx.var = sgtx.var.copy()
all_gtx.obs.loc[:, "lognreads"] = np.log(1 + all_gtx.X.sum(1))
all_gtx.write("data/gtxcombined.h5ad")
