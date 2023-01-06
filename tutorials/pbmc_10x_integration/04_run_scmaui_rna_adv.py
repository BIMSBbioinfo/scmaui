import os
import scanpy as sc
from scmaui.data import load_data, SCMauiDataset
from scmaui.data import combine_modalities
from scmaui.utils import get_model_params
from scmaui.ensembles import EnsembleVAE
from tensorflow.keras.callbacks import CSVLogger

# test adv last layer
outputpath = "./scmaui_pbmc_rna_adv_v1"

datapaths = ["data/gtxcombined.h5ad"]

adatas = load_data(datapaths, ["gtx"])
dataset = SCMauiDataset(adatas, losses=["negbinom"], union=True, adversarial=["sample"])
print(dataset)
params = get_model_params(dataset)
params["nhidden_e"] = 128
params["nlayers_e"] = 2
params["nlayers_d"] = 2
params["nhidden_d"] = 128
params["nlatent"] = 30
print(params)

ensemblevae = EnsembleVAE(params=params)

ensemblevae.fit(dataset, epochs=200, batch_size=32)
ensemblevae.save(outputpath)
ensemblevae.load(outputpath)

latent, _ = ensemblevae.encode(dataset)

# save the latent encoding
latent.to_csv(os.path.join(outputpath, "latent.csv"))

adata = combine_modalities(dataset.adata["input"])
adata = adata
adata.obsm["scmaui-ensemble"] = latent.values

sc.pp.neighbors(adata, n_neighbors=15, use_rep="scmaui-ensemble")
sc.tl.louvain(adata)
sc.tl.umap(adata)
print(adata)
adata.write(os.path.join(outputpath, "analysis.h5ad"), compression="gzip")
print(os.path.join(outputpath, "analysis.h5ad"))
