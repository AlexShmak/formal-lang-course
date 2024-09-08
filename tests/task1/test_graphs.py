from project.graphs import Graph


def test_get_graph_data():
    graph = Graph("travel")
    expected_labels = {
        "type",
        "subClassOf",
        "first",
        "rest",
        "disjointWith",
        "onProperty",
        "someValuesFrom",
        "domain",
        "range",
        "comment",
        "equivalentClass",
        "intersectionOf",
        "differentFrom",
        "hasValue",
        "oneOf",
        "minCardinality",
        "inverseOf",
        "hasPart",
        "hasAccommodation",
        "unionOf",
        "complementOf",
        "versionInfo",
    }

    assert graph.nodes == 131
    assert graph.edges == 277
    for i in graph.labels:
        assert i in expected_labels
