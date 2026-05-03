"""Import smoke tests for public modules and renamed entry points."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "config",
        "compute_prior",
        "eval_ardan",
        "eval_ddpm",
        "eval_fm",
        "inference_ddpm",
        "inference_fm",
        "metrics_ardan",
        "models",
        "train_ddpm",
        "train_fm",
    ],
)
def test_top_level_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name",
    [
        "models.cm_diff_unet",
        "models.ir2red_ddpm",
        "models.red2ir_ddpm",
        "models.ir2red_fm",
        "models.red2ir_fm",
    ],
)
def test_model_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


def test_models_exports_explicit_ddpm_and_fm_names() -> None:
    models = importlib.import_module("models")

    for name in (
        "BidirectionalDDPMUNet",
        "IR2REDDDPMUNet",
        "RED2IRDDPMUNet",
        "IR2REDFMUNet",
        "RED2IRFMUNet",
    ):
        assert hasattr(models, name)


@pytest.mark.parametrize("legacy_name", ["UNet", "IR2REDUNet", "RED2IRUNet"])
def test_models_do_not_export_legacy_unet_names(legacy_name: str) -> None:
    models = importlib.import_module("models")
    assert not hasattr(models, legacy_name)
