import os
import pkg_resources
from scmaui.data import load_data, SCDataset
from scmaui.utils import get_model_params
from scmaui.ensembles import EnsembleVAE


def test_vae_default(tmpdir):

    outputpath = os.path.join(tmpdir.strpath, "output")

    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([peaks], ["peaks"])
    dataset = SCDataset(adatas, ["negbinom"])

    params = get_model_params(dataset)

    ensemble_size = 1
    ensemblevae = EnsembleVAE(params=params, ensemble_size=ensemble_size)

    ensemblevae.fit(dataset)
    latent, latent_list = ensemblevae.encode(dataset)

    assert latent.shape == (100, 10)


def test_vae_pairedintersect(tmpdir):

    outputpath = os.path.join(tmpdir.strpath, "output")

    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    peaks = os.path.join(data_path, "peaks.h5ad")
    gtx = os.path.join(data_path, "gtx.h5ad")

    adatas = load_data([peaks, gtx], ["peaks", "gtx"])
    dataset = SCDataset(adatas, ["negmul", "negbinom"], union=False)

    params = get_model_params(dataset)

    ensemble_size = 1
    ensemblevae = EnsembleVAE(params=params, ensemble_size=ensemble_size)

    ensemblevae.fit(dataset)
    latent, latent_list = ensemblevae.encode(dataset)

    assert latent.shape == (50, 10)


def test_vae_pairedunion(tmpdir):

    outputpath = os.path.join(tmpdir.strpath, "output")

    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    peaks = os.path.join(data_path, "peaks.h5ad")
    gtx = os.path.join(data_path, "gtx.h5ad")

    adatas = load_data([peaks, gtx], ["peaks", "gtx"])
    dataset = SCDataset(adatas, ["negmul", "negbinom"], union=True)

    params = get_model_params(dataset)

    ensemble_size = 1
    ensemblevae = EnsembleVAE(params=params, ensemble_size=ensemble_size)

    ensemblevae.fit(dataset)
    latent, latent_list = ensemblevae.encode(dataset)

    assert latent.shape == (150, 10)


def test_vae_explain_gtx(tmpdir):

    outputpath = os.path.join(tmpdir.strpath, "output")

    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")

    adatas = load_data([gtx], ["gtx"])
    dataset = SCDataset(adatas, ["negbinom"], union=True)

    params = get_model_params(dataset)

    ensemble_size = 1
    ensemblevae = EnsembleVAE(params=params, ensemble_size=ensemble_size)

    ensemblevae.fit(dataset)
    latent, latent_list = ensemblevae.encode(dataset)

    cellids = dataset.adata["input"][0].obs.index.tolist()[:10]
    explanation = ensemblevae.explain(dataset, cellids)

    assert explanation[0].shape == (35300, 10)


def test_vae_explain_multi_I(tmpdir):

    outputpath = os.path.join(tmpdir.strpath, "output")

    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(adatas, ["negbinom", "negmul"], union=True)

    params = get_model_params(dataset)

    ensemble_size = 1
    ensemblevae = EnsembleVAE(params=params, ensemble_size=ensemble_size)

    ensemblevae.fit(dataset)
    latent, latent_list = ensemblevae.encode(dataset)

    cellids = dataset.adata["input"][0].obs.index.tolist()[:10]
    explanation = ensemblevae.explain(dataset, cellids)

    assert explanation[0].shape == (35300, 10)
    assert explanation[1].shape == (95482, 10)


def test_vae_explain_multi_II(tmpdir):

    outputpath = os.path.join(tmpdir.strpath, "output")

    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(adatas, ["negbinom", "negmul"], union=False)

    params = get_model_params(dataset)

    ensemble_size = 1
    ensemblevae = EnsembleVAE(params=params, ensemble_size=ensemble_size)

    ensemblevae.fit(dataset)
    latent, latent_list = ensemblevae.encode(dataset)

    cellids = dataset.adata["input"][0].obs.index.tolist()[:10]
    explanation = ensemblevae.explain(dataset, cellids)

    assert explanation[0].shape == (35300, 10)
    assert explanation[1].shape == (95482, 10)
