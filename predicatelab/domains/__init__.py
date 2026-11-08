"""Domain adapters.

``tsil``, ``tabular`` and ``small_sample`` are executable.  ``mineral`` and
``image`` expose validated descriptor contracts and reserved executor hooks;
their professional algorithms are intentionally not implemented.
"""

from __future__ import annotations

from typing import Any

from predicatelab.core.adapter import DomainAdapter


def create_adapter(config: dict[str, Any]) -> DomainAdapter:
    domain = config.get("run", {}).get("domain") or config.get("domain")
    if domain == "tsil":
        from predicatelab.domains.tsil.adapter import TSILAdapter

        return TSILAdapter(config)
    if domain == "tabular":
        from predicatelab.domains.tabular.adapter import TabularAdapter

        return TabularAdapter(config)
    if domain == "small_sample":
        from predicatelab.domains.small_sample.adapter import SmallSampleAdapter

        return SmallSampleAdapter(config)
    if domain == "mineral":
        from predicatelab.domains.mineral.adapter import MineralAdapter

        return MineralAdapter(config)
    if domain == "image":
        from predicatelab.domains.image.adapter import ImageAdapter

        return ImageAdapter(config)
    raise KeyError(
        f"unknown domain {domain!r}; expected one of tsil, tabular, small_sample, mineral, image"
    )


def executable_domains() -> list[str]:
    return ["tsil", "tabular", "small_sample"]


def reserved_domains() -> list[str]:
    return ["mineral", "image"]
