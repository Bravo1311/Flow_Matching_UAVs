from model.flow_matching_v1.dataset import LandingDataset

ds = LandingDataset()
print(f"Total training examples: {len(ds)}")

history, chunk = ds[0]
print(f"History shape: {history.shape}")   # expect (4, 7)
print(f"Chunk shape: {chunk.shape}")       # expect (8, 4)
print(f"History sample:\n{history}")
print(f"Chunk sample:\n{chunk}")

# spot-check a few random indices too
import random
for i in random.sample(range(len(ds)), 3):
    h, c = ds[i]
    assert h.shape == (4, 7), f"bad history shape at {i}: {h.shape}"
    assert c.shape == (8, 4), f"bad chunk shape at {i}: {c.shape}"
print("Shape checks passed.")