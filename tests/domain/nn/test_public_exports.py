import inspect

import dlkit.domain.nn as domain_nn
import dlkit.gnn as public_gnn
import dlkit.nn as public_nn

# --- Dense FFNN exports ---


def test_varwidth_and_constant_ffnn_exported():
    for ns in (domain_nn, public_nn):
        assert hasattr(ns, "VarWidthFFNN")
        assert hasattr(ns, "FFNN")


def test_scale_equivariant_ffnn_exported():
    for ns in (domain_nn, public_nn):
        assert hasattr(ns, "ScaleEquivariantFFNN")


def test_film_family_exported():
    names = (
        "FiLMBlock",
        "FiLMResidualBlock",
        "FiLMFFNN",
        "FiLMEmbeddedFFNN",
        "VarWidthFiLMFFNN",
        "ScaleEquivariantFiLMFFNN",
        "ScaleEquivariantFiLMEmbeddedFFNN",
        "ScaleEquivariantVarWidthFiLMFFNN",
    )
    for name in names:
        assert hasattr(domain_nn, name), f"{name!r} missing from dlkit.domain.nn"
        assert hasattr(public_nn, name), f"{name!r} missing from dlkit.nn"


def test_removed_se_varwidth_not_exported():
    for ns in (domain_nn, public_nn):
        assert not hasattr(ns, "ScaleEquivariantVarWidthFFNN")
        assert not hasattr(ns, "ScaleEquivariantSimpleVarWidthFFNN")
        assert not hasattr(ns, "ScaleEquivariantSimpleFFNN")


def test_removed_simple_classes_not_exported():
    for ns in (domain_nn, public_nn):
        assert not hasattr(ns, "SimpleFFNN")
        assert not hasattr(ns, "SimpleVarWidthFFNN")


def test_coordinate_spectral_bias_models_exported():
    names = (
        "FourierFeatureNetwork",
        "HashEncodingNetwork",
        "ModifiedMLP",
        "Siren",
        "ScaleEquivariantFourierFeatureNetwork",
        "ScaleEquivariantModifiedMLP",
        "ScaleEquivariantSiren",
    )
    for name in names:
        assert hasattr(domain_nn, name), f"{name!r} missing from dlkit.domain.nn"
        assert hasattr(public_nn, name), f"{name!r} missing from dlkit.nn"


def test_legacy_siren_name_not_exported():
    assert not hasattr(domain_nn, "SirenFFNN")
    assert not hasattr(public_nn, "SirenFFNN")


def test_named_parametric_bases_exported():
    for ns in (domain_nn, public_nn):
        assert hasattr(ns, "EmbeddedParametricFFNN")
        assert hasattr(ns, "EmbeddedSimpleParametricFFNN")


def test_public_namespaces_export_symmetric_constrained_pairs():
    pairs = [
        ("EmbeddedSPDFFNN", "EmbeddedSimpleSPDFFNN"),
        ("SPDFFNN", "SimpleSPDFFNN"),
        ("SPDFactorizedFFNN", "SimpleSPDFactorizedFFNN"),
        ("EmbeddedSPDFactorizedFFNN", "EmbeddedSimpleSPDFactorizedFFNN"),
        ("FactorizedFFNN", "SimpleFactorizedFFNN"),
        (
            "ScaleEquivariantEmbeddedSPDFactorizedFFNN",
            "ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN",
        ),
        ("ScaleEquivariantSPDFFNN", "ScaleEquivariantSimpleSPDFFNN"),
        ("ScaleEquivariantFactorizedFFNN", "ScaleEquivariantSimpleFactorizedFFNN"),
    ]
    for residual_name, plain_name in pairs:
        for ns in (domain_nn, public_nn):
            assert hasattr(ns, residual_name), f"{residual_name!r} missing from {ns.__name__}"
            assert hasattr(ns, plain_name), f"{plain_name!r} missing from {ns.__name__}"


# --- FFNN and VarWidthFFNN accept skip kwarg ---


def test_ffnn_has_skip_param():
    sig = inspect.signature(domain_nn.FFNN.__init__)
    assert "skip" in sig.parameters


def test_varwidth_ffnn_has_skip_param():
    sig = inspect.signature(domain_nn.VarWidthFFNN.__init__)
    assert "skip" in sig.parameters


# --- Graph classes ---


def test_graph_simple_variants_exported_from_graph_module():
    import dlkit.domain.nn.graph as graph_mod

    for name in ("SimpleGATv2Message", "SimpleGATv2Projection", "ScaledSimpleGATv2Projection"):
        assert hasattr(graph_mod, name), f"{name!r} missing from dlkit.domain.nn.graph"


def test_graph_variants_are_reexported_from_graph_namespaces():
    names = (
        "GATv2Message",
        "SimpleGATv2Message",
        "GATv2Projection",
        "SimpleGATv2Projection",
        "ScaledGATv2Projection",
        "ScaledSimpleGATv2Projection",
    )
    for name in names:
        assert not hasattr(domain_nn, name), f"{name!r} should not be exported from dlkit.domain.nn"
        assert not hasattr(public_nn, name), f"{name!r} should not be exported from dlkit.nn"
        assert hasattr(public_gnn, name), f"{name!r} missing from dlkit.gnn"


def test_gatv2_projection_is_a_class():
    import dlkit.domain.nn.graph as graph_mod

    assert isinstance(graph_mod.GATv2Projection, type)
    assert isinstance(graph_mod.ScaledGATv2Projection, type)
    assert isinstance(graph_mod.SimpleGATv2Projection, type)
    assert isinstance(graph_mod.ScaledSimpleGATv2Projection, type)
