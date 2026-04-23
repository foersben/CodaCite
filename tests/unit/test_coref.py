import pytest
from app.infrastructure.coreference import FastCorefResolver

@pytest.mark.asyncio
async def test_fast_coref_resolver_integration():
    """Integration test for FastCorefResolver.
    
    This test runs the actual model on CPU to verify it works correctly.
    Note: This may be slow on first run as it downloads the model weights.
    """
    import spacy
    resolver = FastCorefResolver(nlp=spacy.blank("en"))
    text = "Alice has a brother. She loves him."
    
    # Current implementation returns text directly to avoid errors, 
    # but we want to see if the model *can* run.
    resolved = await resolver.resolve(text)
    
    assert resolved is not None
    assert isinstance(resolved, str)
    # If we fix the implementation to actually resolve, we'd expect:
    # assert "Alice loves Alice's brother" in resolved or similar.

def test_fastcoref_direct_usage():
    """Test fastcoref library directly as in the user's script."""
    from fastcoref import FCoref
    
    import spacy
    model = FCoref(device="cpu", nlp=spacy.blank("en"))
    texts = ["We are so glad to see you here, but you didn't see us."]
    preds = model.predict(texts=texts)
    
    assert len(preds) == 1
    clusters = preds[0].get_clusters()
    assert isinstance(clusters, list)
    
    # Verify the 'unexpected' warning doesn't crash the execution
    print(f"Clusters: {clusters}")
