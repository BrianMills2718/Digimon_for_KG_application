from Core.AgentTools.query_expansion import QueryExpander


def test_query_expansion_preserves_original_and_subject():
    expander = QueryExpander()

    terms = expander.expand_query("What is crystal technology?")

    assert "what is crystal technology?" in terms
    assert "crystal technology" in terms
    assert "crystal" in terms
    assert "technology" in terms


def test_query_expansion_does_not_inject_fixture_specific_knowledge():
    expander = QueryExpander()

    terms = expander.expand_query("What is crystal technology?")

    assert "levitite technology" not in terms
    assert "zorathian empire" not in terms
    assert "great crystal plague" not in terms
