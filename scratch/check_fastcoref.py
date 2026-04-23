from fastcoref import FCoref
import os

model = FCoref(device="cpu")
text = "Alice has a brother. She loves him."
# Try different ways to resolve
print("Testing prediction...")
preds = model.predict(texts=[text])

print(f"Clusters (strings): {preds[0].get_clusters()}")
print(f"Clusters (indices): {preds[0].get_clusters(as_strings=False)}")

# Some versions might have this
try:
    resolved = preds[0].get_resolved_content()
    print(f"Resolved content: {resolved}")
except AttributeError:
    print("No get_resolved_content method")

# Let's try to implement a simple replacement if needed
def simple_resolve(text, clusters):
    # This is complex because of overlapping ranges and multiple mentions
    # But for a test, we just want to see if we CAN do it.
    return text

print("Done")
