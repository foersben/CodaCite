from fastcoref import FCoref

model = FCoref(device="cpu")
preds = model.predict(texts=["We are so glad to see you here, but you didn't see us."])
print(preds[0].get_clusters())
print(preds[0].get_clusters(as_strings=False))
