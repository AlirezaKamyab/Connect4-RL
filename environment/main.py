import pygame
import numpy as np
from env import ConnectFour
from renderer import ConnectFourRenderer


def main():
    pygame.init()
    settings = {
        'screen_width': 1500,
        'screen_height': 900,
        'voffset': 50
    }
    board = ConnectFour(ncols=7, nrows=6, connect=4, render_mode='human', **settings)
    board.reset()

    running = True
    player = 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                entry_pos = board.renderer.position_to_entry(mouse_pos)
                if entry_pos is not None and not board.terminated:
                    _, col = entry_pos
                    if not board.is_valid_action(col):
                        print("Please input a valid action")
                        continue
                    state, rewards, terminated = board.step(col, player=player)
                    print(state)
                    player += 1
                    if player == 3: player = 1
                    print(rewards, 'player turn is', player)
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE] or event.type == pygame.QUIT:
                running = False
            if keys[pygame.K_r]:
                state = board.reset()
        

if __name__ == '__main__':
    main()
