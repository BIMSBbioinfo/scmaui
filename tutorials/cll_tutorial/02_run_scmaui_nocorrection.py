import os
import scanpy as sc
from scmaui.data import load_data, SCMauiDataset
from scmaui.data import combine_modalities
from scmaui.utils import get_model_params
from scmaui.ensembles import EnsembleVAE
from tensorflow.keras.callbacks import CSVLogger


outputpath = './scmaui_cll_model'
ensemble_size = 1

datapaths = ['data/drugs.h5ad', 'data/mrna.h5ad',
             'data/methylation.h5ad', 'data/mutations.h5ad']

adatas = load_data(datapaths, ['drugs', 'mRNA', 'Methyl', 'Mutation'])

dataset = SCMauiDataset(adatas, losses=['mse','mse','mse','binary'], union=True)
params = get_model_params(dataset)
print(params)

ensemblevae = EnsembleVAE(params=params,
                          ensemble_size=ensemble_size)

ensemblevae.fit(dataset, epochs=300, batch_size=16)
ensemblevae.save(outputpath)
ensemblevae.load(outputpath)

latent, _ = ensemblevae.encode(dataset)

# save the latent encoding
latent.to_csv(os.path.join(outputpath, 'latent.csv'))

adata = combine_modalities(dataset.adata['input'])
adata = adata
adata.obsm['scmaui-ensemble'] = latent.values

sc.pp.neighbors(adata, n_neighbors=15, use_rep="scmaui-ensemble")
sc.tl.louvain(adata)
sc.tl.umap(adata)
adata.write(os.path.join(outputpath, "analysis.h5ad"), compression='gzip')

