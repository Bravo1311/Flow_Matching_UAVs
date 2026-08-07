import json
import os
import numpy as np
from model.flow_matching_v1.config import *
import torch
from torch.utils.data import Dataset

def load_and_split_episodes(data_dir, val_fractions=0.1, seed=0):
    import random
    filenames = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    rng = random.Random(seed)
    rng.shuffle(filenames)
    val_count = int(len(filenames) * val_fractions)
    val_files = filenames[:val_count]
    train_files = filenames[val_count:]

    def load(files):
        # returns a list of all steps grouped episode-wise.
        # [ep1:steps, ep2:steps, ...]
        episodes = []
        for fname in files:
            with open(os.path.join(data_dir, fname)) as f:
                episodes.append(json.load(f)["steps"])
        return episodes

    return load(train_files), load(val_files)

class LandingDataset(Dataset):
    """
        Loads all episode JSON filesand builds a flat list of (ep_inx, ts_idx) pairs - one entry per valid training example
    """
    def __init__(self,episode_steps_list, history_len=HISTORY_LEN, chunk_len=CHUNK_LEN):
        self.history_len = history_len   # including current time t
        self.chunk_len = chunk_len
        self.episodes = episode_steps_list
        self.index = []     # list of (ep_idx, t)

        for ep_idx, steps in enumerate(self.episodes):
            T = len(steps)
            for t in range(self.history_len - 1, T - self.chunk_len):
                self.index.append((ep_idx, t))

    def __len__(self):
        return len(self.index)

    # fetches the pose history and action chunks at a given index
    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        steps = self.episodes[ep_idx]

        # --- build history: poses at [t-H+1, ..., t] ---
        history = []
        for i in range(t - self.history_len + 1, t + 1):
            pos = steps[i]["relative_pos"]
            quat = steps[i]["relative_quat"]
            history.append(pos + quat)  # list concat
        history = np.array(history, dtype = np.float32)    # shape (H, 7)

        # --- build action chunk: cmd_vel at [t, ..., t + C -1]
        chunk = []
        for i in range(t, t + self.chunk_len):
            act = steps[i]["cmd_vel"]
            chunk.append(act)
        chunk = np.array(chunk, dtype = np.float32)   # shape (C, 4)
        
        return torch.from_numpy(history), torch.from_numpy(chunk)
            
