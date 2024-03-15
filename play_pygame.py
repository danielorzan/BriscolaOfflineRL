import pygame
from pygame.locals import *
from human_play import load_images
from utils_pygame import draw_cards, draw_cards_result
from decision_transformer import DecisionTransformer
import time
from human_env import CardGameEnvironment
import torch
import os
from copy import deepcopy

def game_play():
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

    eval_chk_pt_dir = ""
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

    rtg_target = 70

    target_shape = (100,180)
    # Load card images
    card_folder = 'mazzo_briscola'
    card_images = load_images(card_folder,target_shape)

    # Initialize Pygame
    pygame.init()

    # Set up the window
    window_width = 800
    window_height = 600
    width = card_images[0].width
    height = card_images[0].height
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Pygame Output")

    # Set up fonts
    font = pygame.font.SysFont(None, 36)
    x1 = (window_width - width*3) // 2 +50
    x2 = x1 + width
    x3 = x2 + width
    y = ((window_height - height*2) // 3)*2 + height

    model.eval()

    close_all = False

    while not close_all:
        # Main loop
        running = True

        env.reset()
        briscola = env.state[-1]

        # Flag to indicate whether a mouse click event has occurred
        mouse_clicked = False
        # Flag to indicate whether part of the code has run already
        part1_run_once = False
        part2_run_once = False

        with torch.no_grad():

            running_rtg = rtg_target / env.rtg_scale
            not_valid_actions = 0

            t = 0
            while running:
                # Event handling
                for event in pygame.event.get():
                    if event.type == QUIT:
                        close_all = True
                        running = False
                    elif event.type == MOUSEBUTTONDOWN:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        # Check if mouse click is within the area of a card
                        if x1 <= mouse_x < x1+width and y <= mouse_y < y+height:  # card 1
                            human_action = 1
                            print("Clicked on card 1")
                            mouse_clicked = True
                        elif x2 <= mouse_x < x2+width and y <= mouse_y < y+height:  # card 2
                            human_action = 2
                            print("Clicked on card 2")
                            mouse_clicked = True
                        elif x3 <= mouse_x < x3+width and y <= mouse_y < y+height:  # card 3
                            human_action = 3
                            print("Clicked on card 3")
                            mouse_clicked = True

                # Clear the screen
                window.fill((255, 255, 255))  # Fill with white

                if env.first: # Agent's turn
                    #print("Your points:", env.points_count[1], "    Cards left:", len(env.cards_left)+1)
                    #print("=" * 60)
                    if not part1_run_once:
                        env.state[1] = 40
                        running_state = env.state_encoding()
                        agent_action, wrongs = env.choose_action(model,t,running_state,running_rtg)
                        not_valid_actions += wrongs
                        draw_cards(env, t, agent_action, env.opp, briscola, card_images, window_width, window_height, window)
                        part1_run_once = True
                        #print("Your turn. Choose your action! 1, 2 or 3")
                    if mouse_clicked:
                        mouse_clicked = False  # Reset the flag
                        env.opp_card = env.opp[human_action-1]
                        old_cards = deepcopy(env.opp)
                        running_reward = env.step(agent_action)
                        #print("=" * 60)
                        draw_cards_result(env, t, [env.opp_card, agent_action], old_cards, briscola, card_images, window_width, window_height, window)
                        running_rtg = running_rtg - (running_reward / env.rtg_scale)
                        if not any(element != 40 for element in env.opp):
                            #print("End of game")
                            env.save_game()
                            if env.points_count[0] < env.points_count[1]:
                                window.fill((255, 255, 255))
                                text_surface = font.render(f"You win with {env.points_count[1]} points!", True, (0, 0, 0))  # Text, antialiasing, color
                                text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))  # Position text
                                # Blit text onto the window
                                window.blit(text_surface, text_rect)
                                pygame.display.flip()
                                time.sleep(3)
                                #print("You win with", env.points_count[1], "points!")
                            elif env.points_count[0] > env.points_count[1]:
                                window.fill((255, 255, 255))
                                text_surface = font.render(f"Agent wins with {env.points_count[0]} points!", True, (0, 0, 0))  # Text, antialiasing, color
                                text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))  # Position text
                                # Blit text onto the window
                                window.blit(text_surface, text_rect)
                                pygame.display.flip()
                                time.sleep(3)
                                #print("Agent wins with", env.points_count[0], "points!")
                            else:
                                window.fill((255, 255, 255))
                                text_surface = font.render("It's a draw!", True, (0, 0, 0))  # Text, antialiasing, color
                                text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))  # Position text
                                # Blit text onto the window
                                window.blit(text_surface, text_rect)
                                pygame.display.flip()
                                time.sleep(3)
                            running = False
                        t += 1
                        part1_run_once = False
                    
                else: # Human player's turn
                    #print("Your points:", env.points_count[1], "    Cards left:", len(env.cards_left)+1)
                    #print("=" * 60)
                    if not part2_run_once:
                        draw_cards(env, t, 40, env.opp, briscola, card_images, window_width, window_height, window)
                        part2_run_once = True
                        #print("Your turn. Choose your action! 1, 2 or 3")
                    if mouse_clicked:
                        mouse_clicked = False  # Reset the flag
                        env.opp_card = env.opp[human_action-1]
                        env.state[1] = deepcopy(env.opp[human_action-1])
                        running_state = env.state_encoding()
                        agent_action, wrongs = env.choose_action(model,t,running_state,running_rtg)
                        not_valid_actions += wrongs
                        old_cards = deepcopy(env.opp)
                        running_reward = env.step(agent_action)
                        #print("=" * 60)
                        draw_cards_result(env, t, [env.opp_card, agent_action], old_cards, briscola, card_images, window_width, window_height, window)
                        running_rtg = running_rtg - (running_reward / env.rtg_scale)
                        # calcualate running rtg and add in placeholder
                        if not any(element != 40 for element in env.opp):
                            #print("End of game")
                            env.save_game()
                            if env.points_count[0] < env.points_count[1]:
                                window.fill((255, 255, 255))
                                text_surface = font.render(f"You win with {env.points_count[1]} points!", True, (0, 0, 0))  # Text, antialiasing, color
                                text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))  # Position text
                                # Blit text onto the window
                                window.blit(text_surface, text_rect)
                                pygame.display.flip()
                                time.sleep(3)
                                #print("You win with", env.points_count[1], "points!")
                            elif env.points_count[0] > env.points_count[1]:
                                window.fill((255, 255, 255))
                                text_surface = font.render(f"Agent wins with {env.points_count[0]} points!", True, (0, 0, 0))  # Text, antialiasing, color
                                text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))  # Position text
                                # Blit text onto the window
                                window.blit(text_surface, text_rect)
                                pygame.display.flip()
                                time.sleep(3)
                                #print("Agent wins with", env.points_count[0], "points!")
                            else:
                                window.fill((255, 255, 255))
                                text_surface = font.render("It's a draw!", True, (0, 0, 0))  # Text, antialiasing, color
                                text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))  # Position text
                                # Blit text onto the window
                                window.blit(text_surface, text_rect)
                                pygame.display.flip()
                                time.sleep(3)
                            running = False
                        t += 1
                        part2_run_once = False
                
                #env.render()
                #print("=" * 60)
        

    # Quit Pygame
    pygame.quit()