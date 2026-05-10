from app.pipelines.generation.router import QueryRouter


def test_router_summarize_intent():
    """Test classification of summarization intent."""
    router = QueryRouter()

    # Positive cases
    assert router.classify_intent("Summarize this document") == "summarize"
    assert router.classify_intent("Give me a summary of the report") == "summarize"
    assert router.classify_intent("tl;dr for this") == "summarize"
    assert router.classify_intent("Provide an overview of the data") == "summarize"
    assert router.classify_intent("Give me a brief outline") == "summarize"
    assert router.classify_intent("what is the brief of this paper?") == "summarize"


def test_router_qa_intent():
    """Test classification of QA intent."""
    router = QueryRouter()

    # Negative cases (should be QA)
    assert router.classify_intent("What is the main topic?") == "qa"
    assert router.classify_intent("Who is the author of this paper?") == "qa"
    assert router.classify_intent("How does the algorithm work?") == "qa"
    assert router.classify_intent("What are the results in table 1?") == "qa"
    assert router.classify_intent("Can you explain the methodology?") == "qa"


def test_router_case_insensitivity():
    """Test case insensitivity of intent classification."""
    router = QueryRouter()
    assert router.classify_intent("SUMMARIZE this") == "summarize"
    assert router.classify_intent("Summary of results") == "summarize"
    assert router.classify_intent("TL;DR please") == "summarize"
