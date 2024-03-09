import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
from datetime import datetime
import h5py

from decision_transformer import DecisionTransformer
from utils import evaluate_on_env
from data_loader import BriscolaLoader
from env import BriscolaEnv

# saves model in this directory
log_dir = "./dt_runs/"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# training and evaluation device
if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)
print("device set to: ", device)

n_training_files = 128 # max 128

hdf5_files = []
for i in range(n_training_files):
    hdf5_files.append(f'data_hdf5_pos_r/data_{i}.h5')

file_games = 8192

def add_epoch(string, letter):
    index = 25
    new_string = string[:index] + letter + string[index + 1:]
    return new_string






max_eval_ep_len = 21    # max len of one evaluation episode
num_eval_ep = 250       # num of evaluation episodes per iteration

batch_size = 16         # training batch size

lr = 1e-4				# learning rate
lr_max = 1              # learning rate after warm up
wt_decay = 1e-3         # weight decay

n_blocks = 2            # num of transformer blocks
embed_dim = 256         # embedding (hidden) dim of transformer
n_heads = 4             # num of transformer heads
dropout_p = 0.1         # dropout probability
initial_temp = 2        # temperature for softmax
final_temp = 1

num_epochs = 8
train_iterations = file_games*len(hdf5_files)//batch_size
warmup_steps = (train_iterations*num_epochs)//3 # warmup steps for lr scheduler

rtg_target = 70
rtg_scale = 120








traj_dataset = BriscolaLoader(hdf5_files, rtg_scale)

traj_data_loader = DataLoader(traj_dataset,
						batch_size=batch_size,
						shuffle=True,
						pin_memory=True,
						drop_last=False
                    )

env = BriscolaEnv(rand=True)

state_dim = traj_dataset.state_dim
act_dim = traj_dataset.act_dim
ep_len = traj_dataset.episode_len

model = DecisionTransformer(
			state_dim=state_dim,
			act_dim=act_dim,
			n_blocks=n_blocks,
			h_dim=embed_dim,
			n_heads=n_heads,
			drop_p=dropout_p,
            temperature=initial_temp
		).to(device)

optimizer = torch.optim.AdamW(
				model.parameters(),
				lr=lr,
				weight_decay=wt_decay
            )

scheduler = torch.optim.lr_scheduler.LambdaLR(
				optimizer,
				lambda steps: min((steps+1)/warmup_steps, lr_max)
            )









start_time = datetime.now().replace(microsecond=0)
start_time_str = start_time.strftime("%d-%H-%M")

save_model_name =  "model_" + start_time_str + "_" + str(0) + "_" + str(lr) + "_" + str(wt_decay) + "_" + str(n_training_files) + "_" + str(batch_size) + "_" + str(embed_dim) + "_" + str(n_blocks) + "_" + str(n_heads) + "_" + str(dropout_p) + ".pt"
save_model_path = os.path.join(log_dir, save_model_name)

save_info_model_path = save_model_path[:-2] + "h5"
save_best_model_path1 = save_model_path[:-3] + "_best_eval.pt"
save_best_model_path2 = save_model_path[:-3] + "_best_errors.pt"
max_score = 93
no_error_seq = 0

print("=" * 60)
print("start time: " + start_time_str)
print("=" * 60)

print("device set to: " + str(device))
print("model save path: " + save_model_path)




for epoch in range(num_epochs):
	model.train()
	traj_dataset.reset() # Reset the dataset at each epoch
	data_iter = iter(traj_data_loader)
	log_action_losses = []
	wrong_acts = []
	evals = []

	# Compute current temperature for the epoch
	current_temperature = initial_temp - (initial_temp - final_temp) * (epoch / (num_epochs-1))
	
	model.temperature = current_temperature

	last_percentage_printed = -1
	for i_train_iter in range(train_iterations):
		timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

		timesteps = timesteps.to(device)								# B x T
		states = states.to(device)										# B x T x 4 x state_dim
		actions = actions.to(device)									# B x T x act_dim
		returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) 		# B x T x 1
		traj_mask = traj_mask.to(device)								# B x T

		action_target = torch.clone(actions).detach().to(device)

		state_preds, action_preds, return_preds, _ = model.forward(
														timesteps=timesteps,
														states=states,
														actions=actions,
														returns_to_go=returns_to_go)

		# only consider non padded elements
		action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
		action_target = action_target.view(-1)[traj_mask.view(-1,) > 0]
		action_loss = F.cross_entropy(action_preds, action_target, reduction='mean')

		#return_loss = F.mse_loss(return_preds, returns_to_go)
		#return_loss *= return_loss_scaling_factor

		total_loss = action_loss #+ return_loss

		optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25) # regularization
		optimizer.step()
		scheduler.step()

		log_action_losses.append(action_loss.detach().cpu().item())
		#log_return_losses.append(return_loss.detach().cpu().item())
		if i_train_iter % 100 == 0 and i_train_iter > 0:
			wins, _, wrongs, _ = evaluate_on_env(model, device, env, rtg_target, rtg_scale, num_eval_ep)
			evals.append(wins)
			wrong_acts.append(sum(wrongs)/len(wrongs))
			# save best validation model (count also the number of straight 0 mistakes)
			if wins >= max_score:
				print("saving best eval model at: " + save_best_model_path1)
				torch.save({
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict(),
				}, save_best_model_path1)
				max_score = wins
			if sum(wrongs) < 1:
				no_error_seq += 1
				if no_error_seq > 1:
					print("saving best error model at: " + save_best_model_path2)
					torch.save({
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'scheduler_state_dict': scheduler.state_dict(),
					}, save_best_model_path2)
			else:
				no_error_seq = 0

		progress_percentage = (i_train_iter + 1) / train_iterations * 100
    
		# Check if the progress percentage has crossed the next integer percentage milestone
		next_percentage = int(progress_percentage)
		if next_percentage % 10 == 0 and next_percentage > last_percentage_printed:
			last_percentage_printed = next_percentage
			_time = datetime.now().replace(microsecond=0)
			_time_str = _time.strftime("%H-%M")
			# Print the progress percentage
			print(f"{next_percentage}% complete at", _time_str)

	# evaluate on env
	# wins, draws, wrong_acts = evaluate_on_env(model, device, env, rtg_target, rtg_scale, num_eval_ep)

	time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
	
	log_str = ("=" * 60 + '\n' +
			"epoch:" + str(epoch) + '\n' +
			"time elapsed: " + time_elapsed + '\n' +
			"action loss: " +  format(action_loss.detach().cpu().item(), ".5f") + '\n' +
			"validation: agent won " + str(wins) + "% of " + str(num_eval_ep) + " games" + '\n'
			)

	print(log_str)

	save_model_path = add_epoch(save_model_path, str(epoch))
	print("saving current model at: ", save_model_path)
	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
	}, save_model_path)

	if not os.path.exists(save_info_model_path):
		with h5py.File(save_info_model_path, 'w') as hf:
			hf.create_dataset('loss', data=log_action_losses ,shape=(train_iterations,), maxshape=(train_iterations*100,), chunks=(train_iterations,), dtype='float') # loss
			hf.create_dataset('wrong_actions', data=wrong_acts, shape=(train_iterations//100,),  maxshape=(train_iterations,), chunks=(train_iterations//100,), dtype='float') # unfeasible actions
			hf.create_dataset('evaluations', data=evals ,shape=(train_iterations//100,), maxshape=(train_iterations,), chunks=(train_iterations//100,), dtype='float') # evaluations
	else:
		with h5py.File(save_info_model_path, 'a') as hf:
			loss = hf['loss']
			wrong_actions = hf['wrong_actions']
			evaluations = hf['evaluations']
			loss.resize((loss.shape[0] + len(log_action_losses),))
			loss[-len(log_action_losses):] = log_action_losses
			wrong_actions.resize((wrong_actions.shape[0] + len(wrong_acts),))
			wrong_actions[-len(wrong_acts):] = wrong_acts
			evaluations.resize((evaluations.shape[0] + len(evals),))
			evaluations[-len(evals):] = evals

print("=" * 60)
print("finished training!")
print("=" * 60)
end_time = datetime.now().replace(microsecond=0)
time_elapsed = str(end_time - start_time)
end_time_str = end_time.strftime("%d-%H-%M")
print("started training at: " + start_time_str)
print("finished training at: " + end_time_str)
print("total training time: " + time_elapsed)
print("minimum loss: " + format(min(log_action_losses), ".5f"))
print("=" * 60)