"""Three-level candidate identity and content-addressed deduplication."""

from predicatelab.core.schema import (
    BoundCandidate,
    CandidateKind,
    FittedInstance,
    PredicateTemplate,
)
from predicatelab.core.store import StateStore


def _template(name="momentum", window=5):
    return PredicateTemplate(
        domain="tsil",
        kind=CandidateKind.NATIVE,
        name=name,
        descriptor={"family": "momentum", "params": {"window": window}},
    )


def test_template_hash_ignores_name_but_tracks_descriptor_and_kind():
    a = _template("a")
    b = _template("b")
    assert a.content_hash == b.content_hash
    assert a.ensure_id().template_id == b.ensure_id().template_id

    c = _template(window=10)
    assert c.content_hash != a.content_hash

    d = PredicateTemplate(
        domain="tsil",
        kind=CandidateKind.SOFT,
        name="a",
        descriptor=a.descriptor,
    )
    assert d.content_hash != a.content_hash


def test_bound_candidate_identity_depends_on_snapshot(tmp_path):
    t = _template().ensure_id()
    base = BoundCandidate(
        template_id=t.template_id,
        template_hash=t.content_hash,
        domain=t.domain,
        kind=t.kind,
        name=t.name,
        descriptor=t.descriptor,
        dataset_id="qlib-cn-csi300",
        snapshot_hash="snap-a",
    ).ensure_id()
    other = base.model_copy(update={"snapshot_hash": "snap-b", "candidate_id": ""}).ensure_id()
    assert base.candidate_id != other.candidate_id


def test_fitted_instance_is_fold_local():
    one = FittedInstance(
        candidate_id="cand",
        fold_id="f2020",
        fitted_params={"threshold": 0.3},
        fit_data_hash="h1",
    ).ensure_id()
    two = FittedInstance(
        candidate_id="cand",
        fold_id="f2021",
        fitted_params={"threshold": 0.7},
        fit_data_hash="h2",
    ).ensure_id()
    assert one.instance_id != two.instance_id


def test_store_deduplicates_templates_and_candidates(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        t1 = _template("one")
        t2 = _template("two")
        id1, created1 = store.add_template(t1)
        id2, created2 = store.add_template(t2)
        assert created1 is True
        assert created2 is False
        assert id1 == id2

        bound = BoundCandidate(
            template_id=id1,
            template_hash=t1.content_hash,
            domain="tsil",
            kind=t1.kind,
            name=t1.name,
            descriptor=t1.descriptor,
            dataset_id="ds",
            snapshot_hash="snap",
        )
        cid1, c1 = store.add_candidate(bound)
        cid2, c2 = store.add_candidate(bound)
        assert c1 is True and c2 is False and cid1 == cid2
