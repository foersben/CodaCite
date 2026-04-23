import traceback
import spacy
from fastcoref import FCoref
import torch

try:
    print("Initializing FCoref with blank nlp...")
    from fastcoref.coref_models.modeling_fcoref import FCorefModel
    if not hasattr(FCorefModel, 'all_tied_weights_keys'):
        FCorefModel.all_tied_weights_keys = property(lambda self: {})
    
    model = FCoref(device="cpu", nlp=spacy.blank("en"))
    print("Prediction...")
    texts = ["We are so glad to see you here, but you didn't see us."]
    preds = model.predict(texts=texts)
    print(f"Success! Clusters: {preds[0].get_clusters()}")
except Exception:
    traceback.print_exc()
