import pygame
from pygame.locals import *
import time

def draw_cards(env, t, agent, card_names, briscola, card_images, window_width, window_height, window):
    width = card_images[agent].width
    height = card_images[agent].height
    #window.fill((255, 255, 255))
    font = pygame.font.SysFont(None, 36)

    if t < 17:
        # Paste deck
        x = window_width // 10 +50 - width
        y = (window_height - height) // 2
        card_surface = pygame.image.fromstring(card_images[40].tobytes(), card_images[40].size, card_images[40].mode)
        window.blit(card_surface, (x, y))

        # Paste briscola
        x = window_width // 10 +50
        y = (window_height - height) // 2
        card_surface = pygame.image.fromstring(card_images[briscola].tobytes(), card_images[briscola].size, card_images[briscola].mode)
        window.blit(card_surface, (x, y))

    # Paste agent's card
    x = (window_width - width) // 2 # Add some space between cards
    y = (window_height - height*2) // 3
    card_surface = pygame.image.fromstring(card_images[agent].tobytes(), card_images[agent].size, card_images[agent].mode)
    window.blit(card_surface, (x, y))

    # Paste player's cards
    y += (y + height)
    x = (window_width - width*3) // 2 +50
    for card in card_names:
        card_surface = pygame.image.fromstring(card_images[card].tobytes(), card_images[card].size, card_images[card].mode)
        window.blit(card_surface, (x, y))
        x += width

    if agent == 40:
        text_surface = font.render("You play first, choose one card", True, (0, 0, 0))  # Text, antialiasing, color
        text_rect = text_surface.get_rect(center=(window_width // 2 +50, window_height // 2))  # Position text
        # Blit text onto the window
        window.blit(text_surface, text_rect)
    else:
        text_surface = font.render("Agent played, choose one card", True, (0, 0, 0))  # Text, antialiasing, color
        text_rect = text_surface.get_rect(center=(window_width // 2 +50, window_height // 2))  # Position text
        # Blit text onto the window
        window.blit(text_surface, text_rect)

    x = window_width // 10 +50
    y = (window_height - height) // 2 -30
    text_surface = font.render(f"Cards left: {len(env.cards_left)}", True, (0, 0, 0))  # Text, antialiasing, color
    text_rect = text_surface.get_rect(center=(x,y))  # Position text
    # Blit text onto the window
    window.blit(text_surface, text_rect)

    x = window_width // 10 +50
    y = (window_height - height) // 2 + height +30
    text_surface = font.render(f"Your points: {env.points_count[1]}", True, (0, 0, 0))  # Text, antialiasing, color
    text_rect = text_surface.get_rect(center=(x,y))  # Position text
    # Blit text onto the window
    window.blit(text_surface, text_rect)
    
    pygame.display.flip()
    return

def draw_cards_result(env, t, played, card_names, briscola, card_images, window_width, window_height, window):
    width = card_images[card_names[0]].width
    height = card_images[card_names[0]].height
    agent = played[0]
    human = played[1]
    #window.fill((255, 255, 255))
    font = pygame.font.SysFont(None, 36)

    if t < 17:
        # Paste deck
        x = window_width // 10 +50 - width
        y = (window_height - height) // 2
        card_surface = pygame.image.fromstring(card_images[40].tobytes(), card_images[40].size, card_images[40].mode)
        window.blit(card_surface, (x, y))

        # Paste briscola
        x = window_width // 10 +50
        y = (window_height - height) // 2
        card_surface = pygame.image.fromstring(card_images[briscola].tobytes(), card_images[briscola].size, card_images[briscola].mode)
        window.blit(card_surface, (x, y))

    # Paste agent's card
    x = (window_width - width) // 2 # Add some space between cards
    y = (window_height - height*2) // 3
    card_surface = pygame.image.fromstring(card_images[human].tobytes(), card_images[human].size, card_images[human].mode)
    window.blit(card_surface, (x, y))

    # Paste human's card
    x = (window_width - width) // 2 + width + 10 # Add some space between cards
    y = (window_height - height*2) // 3
    card_surface = pygame.image.fromstring(card_images[agent].tobytes(), card_images[agent].size, card_images[agent].mode)
    window.blit(card_surface, (x, y))

    # Paste player's cards
    x = (window_width - width*3) // 2 +50
    y += (y + height)
    for card in card_names:
        if card != human:
            card_surface = pygame.image.fromstring(card_images[card].tobytes(), card_images[card].size, card_images[card].mode)
            window.blit(card_surface, (x, y))
        x += width

    if env.first:
        text_surface = font.render(f"Agent wins the trick with {env.current_points} points.", True, (0, 0, 0))  # Text, antialiasing, color
        text_rect = text_surface.get_rect(center=(window_width // 2 +100, window_height // 2))  # Position text
        # Blit text onto the window
        window.blit(text_surface, text_rect)
        #print("Agent won the trick with", env.current_points, "points.")
    else:
        text_surface = font.render(f"You win the trick with {env.current_points} points.", True, (0, 0, 0))  # Text, antialiasing, color
        text_rect = text_surface.get_rect(center=(window_width // 2 +50, window_height // 2))  # Position text
        # Blit text onto the window
        window.blit(text_surface, text_rect)
        #print("You won the trick with", env.current_points, "points.")
    
    pygame.display.flip()
    time.sleep(3)
    return
