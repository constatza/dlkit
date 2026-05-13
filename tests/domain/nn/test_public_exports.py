import inspect

import dlkit.domain.nn as domain_nn
import dlkit.nn as public_nn

# --- Dense FFNN pairs ---


def test_original_dense_pairs_still_exported():
    for ns in (domain_nn, public_nn):
        assert hasattr(ns, "FeedForwardNN")
        assert hasattr(ns, "SimpleFeedForwardNN")
        assert hasattr(ns, "ConstantWidthFFNN")
        assert hasattr(ns, "ConstantWidthSimpleFFNN")


def test_se_variable_width_pair_exported():
    for ns in (domain_nn, public_nn):
        assert hasattr(ns, "ScaleEquivariantFeedForwardNN")
        assert hasattr(ns, "ScaleEquivariantSimpleFeedForwardNN")


def test_scale_equivariant_ffnn_not_exported():
    assert not hasattr(domain_nn, "ScaleEquivariantFFNN")
    assert not hasattr(public_nn, "ScaleEquivariantFFNN")


def test_coordinate_spectral_bias_models_exported():
    names = (
        "FourierFeatureNetwork",
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
        assert hasattr(ns, "ConstantWidthParametricFFNN")
        assert hasattr(ns, "ConstantWidthSimpleParametricFFNN")
        assert hasattr(ns, "EmbeddedParametricFFNN")
        assert hasattr(ns, "EmbeddedSimpleParametricFFNN")


def test_public_namespaces_export_symmetric_constrained_pairs():
    pairs = [
        ("ConstantWidthFactorizedFFNN", "ConstantWidthSimpleFactorizedFFNN"),
        ("EmbeddedSPDFFNN", "EmbeddedSimpleSPDFFNN"),
        (
            "ScaleEquivariantEmbeddedSPDFactorizedFFNN",
            "ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN",
        ),
    ]
    for residual_name, plain_name in pairs:
        for ns in (domain_nn, public_nn):
            assert hasattr(ns, residual_name), f"{residual_name!r} missing from {ns.__name__}"
            assert hasattr(ns, plain_name), f"{plain_name!r} missing from {ns.__name__}"


# --- No public residual: bool on targeted constructors ---

_TARGETED_CLASSES = [
    "ConstantWidthParametricFFNN",
    "ConstantWidthSimpleParametricFFNN",
    "EmbeddedParametricFFNN",
    "EmbeddedSimpleParametricFFNN",
    "ScaleEquivariantFeedForwardNN",
    "ScaleEquivariantSimpleFeedForwardNN",
]


def test_no_targeted_ffnn_class_has_residual_param():
    for name in _TARGETED_CLASSES:
        cls = getattr(domain_nn, name, None)
        if cls is None:
            continue
        sig = inspect.signature(cls.__init__)
        assert "residual" not in sig.parameters, f"{name} still has public residual param"


# --- Graph classes ---


def test_graph_simple_variants_exported_from_graph_module():
    import dlkit.domain.nn.graph as graph_mod

    for name in ("SimpleGATv2Message", "SimpleGATv2Projection", "ScaledSimpleGATv2Projection"):
        assert hasattr(graph_mod, name), f"{name!r} missing from dlkit.domain.nn.graph"


def test_graph_variants_are_reexported_from_top_level_namespaces():
    names = (
        "GATv2Message",
        "SimpleGATv2Message",
        "GATv2Projection",
        "SimpleGATv2Projection",
        "ScaledGATv2Projection",
        "ScaledSimpleGATv2Projection",
    )
    for name in names:
        assert hasattr(domain_nn, name), f"{name!r} missing from dlkit.domain.nn"
        assert hasattr(public_nn, name), f"{name!r} missing from dlkit.nn"


def test_gatv2_projection_is_a_class():
    import dlkit.domain.nn.graph as graph_mod

    assert isinstance(graph_mod.GATv2Projection, type)
    assert isinstance(graph_mod.ScaledGATv2Projection, type)
    assert isinstance(graph_mod.SimpleGATv2Projection, type)
    assert isinstance(graph_mod.ScaledSimpleGATv2Projection, type)


_TARGETED_GRAPH_CLASSES = [
    "GATv2Message",
    "SimpleGATv2Message",
    "GATv2Projection",
    "SimpleGATv2Projection",
    "ScaledGATv2Projection",
    "ScaledSimpleGATv2Projection",
]


def test_no_targeted_graph_class_has_residual_param():
    for name in _TARGETED_GRAPH_CLASSES:
        cls = getattr(domain_nn, name, None)
        if cls is None:
            continue
        sig = inspect.signature(cls.__init__)
        assert "residual" not in sig.parameters, f"{name} still has public residual param"
