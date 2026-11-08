"""Reserved mineral / image extension contracts.

V1.0 deliberately ships only schemas and validators for these domains; every
execution path must raise DomainNotImplemented rather than return fake results.
"""

import pytest

from predicatelab.core.adapter import DomainNotImplemented
from predicatelab.domains import executable_domains, reserved_domains
from predicatelab.domains.image.adapter import ImageAdapter
from predicatelab.domains.mineral.adapter import MineralAdapter


def _cfg(domain):
    return {
        "run": {"domain": domain, "artifacts_dir": "/tmp/predicatelab-test"},
        "data": {"provider_uri": "/tmp/none", "tsil_root": "/tmp/none"},
    }


def test_registry_separates_executable_and_reserved_domains():
    assert set(executable_domains()) == {"tsil", "tabular"}
    assert set(reserved_domains()) == {"mineral", "image"}


def test_reserved_domain_validation_is_structural_only():
    mineral = MineralAdapter(_cfg("mineral"))
    image = ImageAdapter(_cfg("image"))

    mineral.validate_descriptor("statistical", {
        "family": "geochemical_anomaly", "variable": "Cu_ppm",
        "transform": "log", "threshold_quantile": 0.9,
    })
    image.validate_descriptor("statistical", {
        "family": "color_statistic", "channel": "r", "statistic": "mean",
    })

    with pytest.raises(ValueError):
        mineral.validate_descriptor("statistical", {"family": "not_a_family"})
    with pytest.raises(ValueError):
        image.validate_descriptor("soft", {
            "family": "soft_visual_predicate", "threshold": 1.5,
            "margin": 0.1, "concept": "smooth",
        })


@pytest.mark.parametrize("adapter_cls", [MineralAdapter, ImageAdapter])
def test_reserved_execution_raises(adapter_cls):
    name = "mineral" if adapter_cls is MineralAdapter else "image"
    adapter = adapter_cls(_cfg(name))
    with pytest.raises(DomainNotImplemented):
        adapter.snapshot({})
    with pytest.raises(DomainNotImplemented):
        adapter.build_protocol({})
    with pytest.raises(DomainNotImplemented):
        adapter.evaluation
    with pytest.raises(DomainNotImplemented):
        adapter.run_trial(None)
