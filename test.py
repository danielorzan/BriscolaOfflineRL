# testing with different target reward

import torch

import os
import h5py

from decision_transformer import DecisionTransformer
from utils import evaluate_on_env
from env import BriscolaEnv

# training and evaluation device
if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)
print("device set to: ", device)

state_dim = 40
act_dim = 41
n_blocks = 2            # num of transformer blocks
embed_dim = 256         # embedding (hidden) dim of transformer
n_heads = 4             # num of transformer heads
dropout_p = 0.1         # dropout probability

rtg_scale = 120

eval_chk_pt_dir = "dt_runs/"
eval_chk_pt_name = "model_03-23-05_7_0.0001_0.001_128_16_256_2_4_0.1_best_eval.pt"

eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)

eval_model = DecisionTransformer(
			state_dim=state_dim,
			act_dim=act_dim,
			n_blocks=n_blocks,
			h_dim=embed_dim,
			n_heads=n_heads,
			drop_p=dropout_p,
		).to(device)

env = BriscolaEnv()

# Load the model state dict
checkpoint = torch.load(eval_chk_pt_path, map_location=device)
eval_model.load_state_dict(checkpoint['model_state_dict'])

print("model loaded from: " + eval_chk_pt_path)

all_wins = []
all_errors = []
for rtg_target in range(61,120):
    wins, _, wrong_acts, _ = evaluate_on_env(eval_model, device, env, rtg_target, rtg_scale, 1000)
    all_wins.append(wins)
    all_errors.append((sum(wrong_acts)/len(wrong_acts))/20*100)
    
with h5py.File('./returns_to_go.h5', 'w') as hf:
	hf.create_dataset('all_wins', data=all_wins, dtype='int')
	hf.create_dataset('all_errors', data=all_errors, dtype='float')