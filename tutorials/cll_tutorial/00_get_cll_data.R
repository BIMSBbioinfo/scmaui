#if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
#
#BiocManager::install("MOFAdata")

library(MOFAdata)
data("CLL_data")
data("CLL_covariates")

write.table(CLL_covariates, "data/cll_covariates.csv", quote=FALSE, sep=",")

write.table(CLL_data$Drugs, "data/drugs.csv", quote=FALSE, sep=",")

write.table(CLL_data$Methylation, "data/methylation.csv", quote=FALSE, sep=",")
write.table(CLL_data$mRNA, "data/mrna.csv", quote=FALSE, sep=",")
write.table(CLL_data$Mutations, "data/mutations.csv", quote=FALSE, sep=",")

