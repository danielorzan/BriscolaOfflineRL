import numpy as np
import torch

def one_hot_encoding(pos):
    vec = np.zeros(40, dtype=float)
    if isinstance(pos, list):
      for p in pos:
        if p != 40:
          vec[p] = 1.0
    else:
       if pos != 40:
          vec[pos] = 1.0
    return vec

def evaluate_on_env(model, device, env, rtg_target, rtg_scale,
                    num_eval_ep=100):

    max_test_ep_len = 21
    eval_batch_size = 1  # required for forward pass

    results = [0,0,0] # win, draw, lose
    wrong_acts = []

    state_dim = 40

    # same as timesteps used for training the transformer
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():
        game_attention_weights = []
        for _ in range(num_eval_ep):
            total_reward = 0

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len),
                                dtype=torch.int64, device=device)

            states = torch.zeros((eval_batch_size, max_test_ep_len, 4, state_dim),
                                dtype=torch.float32, device=device)

            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)

            # init episode
            running_state_list = env.reset()
            running_state = []
            for i in range(4):
              running_state.append(one_hot_encoding(running_state_list[i]))

            running_reward = 0
            running_rtg = rtg_target / rtg_scale
            not_valid_actions = 0
            
            for t in range(20):
                # add state in placeholder and normalize
                states[0,t,:,:] = torch.tensor(running_state, dtype=torch.float32).to(device)
                # states[0, t] = (states[0, t] - state_mean) / state_std

                # calcualate running rtg and add in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                _, act_preds, _, attention_weights = model.forward(timesteps[:,:t+1],
                                            states[:,:t+1,:,:],
                                            actions[:,:t+1],
                                            rewards_to_go[:,:t+1])

                game_attention_weights.append(attention_weights)

                act = act_preds[0,-1].detach()
                action = np.argmax(act.cpu().numpy()) # max out of 41 long array

                running_state_list, running_reward, n = env.step(action)
                running_state = []
                for i in range(4):
                    running_state.append(one_hot_encoding(running_state_list[i]))

                # add action in placeholder
                actions[0, t] = action

                total_reward += running_reward
                not_valid_actions += n

            if total_reward > 60:
              results[0] += 1
            elif total_reward < 60:
              results[2] += 1
            else:
              results[1] += 1
            wrong_acts.append(not_valid_actions)
            if not_valid_actions > 0:
               env.wrong_games.append(env.game)

    return results[0]/num_eval_ep*100, results[1]/num_eval_ep*100, wrong_acts, game_attention_weights