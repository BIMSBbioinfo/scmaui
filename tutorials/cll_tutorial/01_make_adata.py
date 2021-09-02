import pandas as pd
import numpy as np
from anndata import AnnData
import scanpy as sc
from scipy.stats import zscore

covariates = pd.read_csv("data/cll_covariates.csv")

drugs = pd.read_csv("data/drugs.csv")

drugs = drugs.T
def make_mask(df):
    return (~pd.isna(drugs.values.sum(1))*1).reshape(-1,1)

adrugs = AnnData(zscore(np.log(drugs.values), axis=1, nan_policy='omit'),
                 obs=covariates.loc[drugs.index,:],
                 var=pd.DataFrame(index=drugs.columns),
                 obsm={'mask': make_mask(drugs)})

adrugs.write("data/drugs_zscore.h5ad")

adrugs = AnnData(np.log(drugs.values),
                 obs=covariates.loc[drugs.index,:],
                 var=pd.DataFrame(index=drugs.columns),
                 obsm={'mask': make_mask(drugs)})

adrugs.write("data/drugs.h5ad")

mrna = pd.read_csv("data/mrna.csv")
mrna = mrna.T

amrna = AnnData(zscore(np.log(mrna.values), axis=1, nan_policy='omit'),
                obs=covariates.loc[mrna.index,:],
                var=pd.DataFrame(index=mrna.columns),
                 obsm={'mask': make_mask(mrna)})
amrna.write("data/mrna_zscore.h5ad")

amrna = AnnData(np.log(mrna.values),
                obs=covariates.loc[mrna.index,:],
                var=pd.DataFrame(index=mrna.columns),
                 obsm={'mask': make_mask(mrna)})
amrna.write("data/mrna.h5ad")

methyl = pd.read_csv("data/methylation.csv")
methyl = methyl.T
amethyl = AnnData(zscore(methyl.values, axis=1, nan_policy='omit'),
                  obs=covariates.loc[methyl.index,:],
                  var=pd.DataFrame(index=methyl.columns),
                 obsm={'mask': make_mask(methyl)})
amethyl.write("data/methylation_zscore.h5ad")

amethyl = AnnData(methyl.values,
                  obs=covariates.loc[methyl.index,:],
                  var=pd.DataFrame(index=methyl.columns),
                 obsm={'mask': make_mask(methyl)})
amethyl.write("data/methylation.h5ad")

mutations = pd.read_csv("data/mutations.csv")
mutations = mutations.T
mutations = AnnData(mutations.values,
                    obs=covariates.loc[mutations.index,:],
                    var=pd.DataFrame(index=mutations.columns),
                 obsm={'mask': make_mask(mutations)})
mutations.write("data/mutations.h5ad")

combined = sc.concat([amrna, amethyl, adrugs], axis=1)
combined.write("data/merged_rna_methyl_drugs.h5ad")
#print(drugs)
#print(mrna)
#print(methyl)
#print(mutations)
print(combined)
