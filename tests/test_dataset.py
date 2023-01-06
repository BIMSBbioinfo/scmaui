import pkg_resources
import os
from scmaui.data import load_data
from scmaui.data import SCDataset
from scmaui.data import get_covariates


def test_singledataset():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")

    adatas = load_data([gtx], ["gtx"])
    dataset = SCDataset(adatas, losses=["mse"])
    assert dataset.size() == 100
    assert dataset.modalities() == (["gtx"], ["gtx"])
    assert dataset.shapes() == {"inputdims": [35300], "outputdims": [35300]}
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_singledataset_conditional():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([peaks], ["peaks"])
    dataset = SCDataset(adatas, losses=["mse"], conditional=["logreads", "sample"])
    assert dataset.size() == 100
    assert dataset.modalities() == (["peaks"], ["peaks"])
    assert dataset.shapes() == {"inputdims": [95482], "outputdims": [95482]}
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_singledataset_adversarial():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([peaks], ["peaks"])
    dataset = SCDataset(adatas, losses=["mse"], adversarial=["logreads", "sample"])
    assert dataset.size() == 100
    assert dataset.modalities() == (["peaks"], ["peaks"])
    assert dataset.shapes() == {"inputdims": [95482], "outputdims": [95482]}
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_paireddataset_intersect():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(adatas, losses=["mse"] * 2, union=False)
    assert dataset.size() == 50
    assert dataset.modalities() == (["gtx", "peaks"], ["gtx", "peaks"])
    assert dataset.shapes() == {
        "inputdims": [35300, 95482],
        "outputdims": [35300, 95482],
    }
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_paireddataset_union():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(adatas, losses=["mse"] * 2, union=True)
    assert dataset.size() == 150
    assert dataset.modalities() == (["gtx", "peaks"], ["gtx", "peaks"])
    assert dataset.shapes() == {
        "inputdims": [35300, 95482],
        "outputdims": [35300, 95482],
    }
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_paireddataset_intersect_conditional():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(
        adatas, losses=["mse"] * 2, union=False, conditional=["logreads", "sample"]
    )
    assert dataset.size() == 50
    assert dataset.modalities() == (["gtx", "peaks"], ["gtx", "peaks"])
    assert dataset.shapes() == {
        "inputdims": [35300, 95482],
        "outputdims": [35300, 95482],
    }
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_paireddataset_union_conditional():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(
        adatas, losses=["mse"] * 2, union=True, conditional=["logreads", "sample"]
    )
    assert dataset.size() == 150
    assert dataset.modalities() == (["gtx", "peaks"], ["gtx", "peaks"])
    assert dataset.shapes() == {
        "inputdims": [35300, 95482],
        "outputdims": [35300, 95482],
    }
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_paireddataset_intersect_adversarial():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(
        adatas, losses=["mse"] * 2, union=False, adversarial=["logreads", "sample"]
    )
    assert dataset.size() == 50
    assert dataset.modalities() == (["gtx", "peaks"], ["gtx", "peaks"])
    assert dataset.shapes() == {
        "inputdims": [35300, 95482],
        "outputdims": [35300, 95482],
    }
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_paireddataset_union_adversarial():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(
        adatas, losses=["mse", "mse"], union=True, adversarial=["logreads", "sample"]
    )
    assert dataset.size() == 150
    assert dataset.modalities() == (["gtx", "peaks"], ["gtx", "peaks"])
    assert dataset.shapes() == {
        "inputdims": [35300, 95482],
        "outputdims": [35300, 95482],
    }
    assert dataset.adversarial_config()
    assert dataset.conditional_config()


def test_covariates():

    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(
        adatas, losses=["mse"] * 2, union=True, adversarial=["logreads", "sample"]
    )
    data, dtype = get_covariates(dataset.adata["input"], ["logreads", "sample"])


def test_subset():
    data_path = pkg_resources.resource_filename("scmaui", "resources/")
    gtx = os.path.join(data_path, "gtx.h5ad")
    peaks = os.path.join(data_path, "peaks.h5ad")

    adatas = load_data([gtx, peaks], ["gtx", "peaks"])
    dataset = SCDataset(
        adatas, losses=["mse"] * 2, union=True, adversarial=["logreads", "sample"]
    )

    first_indices = dataset.adata["input"][0].obs.index.tolist()[:10]
    print(first_indices)

    subdataset = dataset.subset(first_indices)

    assert subdataset.size() == 10
    assert subdataset.modalities() == (["gtx", "peaks"], ["gtx", "peaks"])
    assert subdataset.shapes() == {
        "inputdims": [35300, 95482],
        "outputdims": [35300, 95482],
    }
    assert subdataset.adversarial_config()
    assert subdataset.conditional_config()
