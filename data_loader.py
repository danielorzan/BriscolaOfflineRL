import torch
from torch.utils.data import Dataset
import random
import h5py


class BriscolaLoader(Dataset):
    def __init__(self, file_paths, rtg_scale):
        self.episode_len = 21
        self.state_dim = 40
        self.act_dim = 41
        self.rtg_scale = rtg_scale
        self.n_games = 8192

        self.file_paths = file_paths
        self.current_file_index = 0
        self.current_data_index = 0

        # Shuffle the order of files
        random.shuffle(self.file_paths)

        # Shuffle the indices of data samples
        self.data_indices = list(range(self.n_games))
        random.shuffle(self.data_indices)

    def load_next_file_data(self):
        self.current_file_index += 1
        if self.current_file_index >= len(self.file_paths):
            return None  # No more files to load
        data = self.load_data_from_file()
        return data
    
    def load_data_from_file(self):
        idx = self.current_data_index
        with h5py.File(self.file_paths[self.current_file_index], 'r') as hf:
            states = torch.tensor(hf['array1'][self.data_indices[idx]], dtype=torch.float32).view(21, 4, 40)
            actions = torch.tensor(hf['array2'][self.data_indices[idx]], dtype=torch.int64)
            returns_to_go = torch.tensor(hf['array3'][self.data_indices[idx]], dtype=torch.float32)/self.rtg_scale
            timesteps = torch.tensor(hf['array4'][self.data_indices[idx]])
            traj_mask = torch.tensor(hf['array5'][self.data_indices[idx]])
        
        return [timesteps, states, actions, returns_to_go, traj_mask]
    
    def reset(self):
        self.current_file_index = 0
        self.current_data_index = 0

        # Shuffle the order of files again for the next epoch
        random.shuffle(self.file_paths)

        # Shuffle the indices of data samples
        random.shuffle(self.data_indices)

    def __len__(self):
        return self.n_games * len(self.file_paths)

    def __getitem__(self, idx):
        # Check if current file data is exhausted, load next file if needed
        if self.current_data_index >= 8192:
            self.current_data_index = 0
            self.current_file_data = self.load_next_file_data()
        # Get data from the current file
        else:
            self.current_file_data = self.load_data_from_file()

        if self.current_file_data is None:
            raise IndexError("Index out of range")

        item = self.current_file_data
        self.current_data_index += 1
        return item[0], item[1], item[2], item[3], item[4]