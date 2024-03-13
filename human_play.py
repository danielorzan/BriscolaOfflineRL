from decision_transformer import DecisionTransformer
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, clear_output
import time
from human_env import CardGameEnvironment
import torch
import os
from copy import deepcopy

# Function to load images from a folder
def load_images(folder_path, target_shape):
    images = {}
    i = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            path = os.path.join(folder_path, filename)
            image = Image.open(path)
            image = image.resize(target_shape)
            images[i] = image
            i += 1
    return images

# Draw card images
def draw_cards(agent, card_names, briscola, card_images):
    print(' '*7,'briscola',' '*11,'agent')
    width = card_images[agent].width
    result_width = width*3
    max_height = max(card_images[card].height for card in card_names)
    result_height = max_height * 2
    result = Image.new('RGB', (result_width, result_height))

    # Paste images onto the blank image for the first line
    x = (result_width - width*2) // 3  # Centering the cards
    y = 0
    result.paste(card_images[briscola], (x, y))
    x += (x + width)  # Add some space between cards
    result.paste(card_images[agent], (x, y))

    # Paste images onto the blank image for the second line
    y = max_height
    x = 0
    for card in card_names:
        result.paste(card_images[card], (x, y))
        x += card_images[card].width

    display(result)
    print(' '*7,'1',' '*11,'2',' '*11,'3')
    return

def draw_cards_result(card_names, card_images):
    width = card_images[card_names[0]].width
    result_width = width*2
    max_height = max(card_images[card].height for card in card_names)
    result = Image.new('RGB', (result_width, max_height))

    x = 0
    for card in card_names:
        result.paste(card_images[card], (x, 0))
        x += card_images[card].width

    # Display the result inline in the notebook
    display(result)

    # Wait for 3 seconds
    time.sleep(3)

    # Clear the output
    clear_output(wait=True)

def human_input():
    while True:
        # Use a text input field to get user input
        user_input = input("Enter a number (1, 2, or 3): ")

        # Convert the input to an integer and check if it's 1, 2, or 3
        try:
            user_number = int(user_input)
            if user_number in [1, 2, 3]:
                print("You entered:", user_number)
                break  # Exit the loop if the input is valid
            else:
                print("Invalid input. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    return user_number

def game_loop():
    env = CardGameEnvironment()
    if torch.cuda.is_available():
        device_name = 'cuda'
    else:
        device_name = 'cpu'
    device = torch.device(device_name)
    #print("device set to: ", device)

    env.device = device

    act_dim = 41
    n_blocks = 2            # num of transformer blocks
    embed_dim = 256         # embedding (hidden) dim of transformer
    n_heads = 4             # num of transformer heads
    dropout_p = 0.1         # dropout probability

    eval_chk_pt_dir = "./"
    eval_chk_pt_name = "model_03-23-05_7_0.0001_0.001_128_16_256_2_4_0.1_best_eval.pt"

    eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)

    model = DecisionTransformer(
                state_dim=env.state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                n_heads=n_heads,
                drop_p=dropout_p,
            ).to(device)

    # Load the model state dict
    checkpoint = torch.load(eval_chk_pt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    #print("model loaded from: " + eval_chk_pt_path)
    #print("=" * 60)

    done = False
    rtg_target = 70
    env.reset()

    target_shape = (100,180)
    # Load card images
    card_folder = './mazzo_briscola'
    card_images = load_images(card_folder,target_shape)
    briscola = env.state[-1]

    model.eval()

    print("The game is starting.")
    print("To choose an action press 1, 2 or 3 and then press enter.")
    # Wait for 3 seconds
    time.sleep(5)

    # Clear the output
    clear_output(wait=True)

    with torch.no_grad():

        running_rtg = rtg_target / env.rtg_scale
        not_valid_actions = 0

        t = 0
        while not done:
            if env.first: # Agent's turn
                print("Your points:", env.points_count[1], "    Cards left:", len(env.cards_left)+1)
                print("=" * 60)
                env.state[1] = 40
                running_state = env.state_encoding()
                agent_action, wrongs = env.choose_action(model,t,running_state,running_rtg)
                not_valid_actions += wrongs
                draw_cards(agent_action, env.opp, briscola, card_images)
                #print("Your turn. Choose your action! 1, 2 or 3")
                print('\n\n\n\n\n\n\n\n\n')
                human_action = human_input()
                env.opp_card = env.opp[human_action-1]
                running_reward = env.step(agent_action)
                print("=" * 60)
                if env.first:
                    print("Agent won the trick with", env.current_points, "points.")
                else:
                    print("You won the trick with", env.current_points, "points.")
                draw_cards_result([agent_action, env.opp_card], card_images)
                running_rtg = running_rtg - (running_reward / env.rtg_scale)
                if not any(element != 40 for element in env.opp):
                    print("End of game")
                    env.save_game()
                    if env.points_count[0] < env.points_count[1]:
                        print("Human player wins!")
                    else:
                        print("Agent wins!")
                    done = True
                
            else: # Human player's turn
                print("Your points:", env.points_count[1], "    Cards left:", len(env.cards_left)+1)
                print("=" * 60)
                draw_cards(40, env.opp, briscola, card_images)
                #print("Your turn. Choose your action! 1, 2 or 3")
                print('\n\n\n\n\n\n\n\n\n')
                human_action = human_input()
                env.opp_card = env.opp[human_action-1]
                env.state[1] = deepcopy(env.opp[human_action-1])
                running_state = env.state_encoding()
                agent_action, wrongs = env.choose_action(model,t,running_state,running_rtg)
                not_valid_actions += wrongs
                running_reward = env.step(agent_action)
                print("=" * 60)
                if env.first:
                    print("Agent won the trick with", env.current_points, "points.")
                else:
                    print("You won the trick with", env.current_points, "points.")
                draw_cards_result([env.opp_card, agent_action], card_images)
                running_rtg = running_rtg - (running_reward / env.rtg_scale)
                # calcualate running rtg and add in placeholder
                if not any(element != 40 for element in env.opp):
                    print("End of game")
                    env.save_game()
                    if env.points_count[0] < env.points_count[1]:
                        print("You win!")
                    else:
                        print("Agent wins!")
                    done = True
            
            #env.render()
            t += 1
            #print("=" * 60)

    return not_valid_actions