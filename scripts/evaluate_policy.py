import torch
from model.flow_matching_v1.transformer import FlowMatchingTransformer

model = FlowMatchingTransformer(pose_dim=7, action_dim=4, history_len=4, chunk_len=8)
B = 16
noisy_actions = torch.randn(B, 8, 4)
t = torch.rand(B)
history = torch.randn(B, 4, 7)

out = model(noisy_actions, t, history)
print(out.shape)  # expect (16, 8, 4)