import numpy as np
from entry import Entry
import pygame


class ConnectFourRenderer:
    def __init__(
            self, 
            state:np.array, 
            screen_width:int=800, 
            screen_height:int=600, 
            *, 
            background:tuple=(255, 255, 255),
            empty_color:tuple=(0, 0, 0), 
            color1:tuple=(122, 150, 10), 
            color2:tuple=(200, 10, 100),
            hpadding:int=None,
            vpadding:int=None,
            voffset:int=50,
            hoffset:int=0,
            width:int=2
        ):
        self.state = state
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.width = width
        
        # setup colors
        self.background = background
        self.empty_colors = empty_color
        self.color1 = color1
        self.color2 = color2

        # entries
        self.entries = None

        # Create board
        self.hpadding = hpadding if hpadding is not None else int(screen_width * 0.01)
        self.vpadding = vpadding if vpadding is not None else int(screen_height * 0.01) 
        self.voffset = voffset
        self.hoffset = hoffset

        if self.state is not None:
            self.clear_screen()
            self.create_board()


    def clear_screen(self):
        self.screen.fill(self.background)

    def update_state(self, state:np.array):
        self.state = state
        self.create_board()

    def position_to_entry(self, pos:tuple):
        rows, columns = self.state.shape
        for i in range(rows):
            for j in range(columns):
                entry = self.entries[i][j]
                if entry.on_entry(pos):
                    return (entry.row, entry.col)


    def create_board(
        self
    ):
        rows, columns = self.state.shape
        screen = self.screen
        screen_width = screen.get_width()
        screen_height = screen.get_height()

        self.entries = [[None for _ in range(columns)] for _ in range(rows)]

        circle_radius = min((screen_width - self.hpadding - self.hoffset) / columns, (screen_height - self.vpadding - self.voffset) / rows) / 2

        per_circle_hpadding = self.hpadding // columns
        per_circle_vpadding = self.vpadding // rows

        max_width = (columns - 1) * 2 * circle_radius + self.hpadding + self.hoffset
        width_center_alignment = (screen_width - max_width) // 2

        max_height = (rows - 1) * 2 * circle_radius + self.vpadding + self.voffset
        height_center_alignment = (screen_height - max_height) // 2

        for row in range(rows):
            for column in range(columns):
                hpos = column * 2 * circle_radius + column * per_circle_hpadding + self.hoffset + width_center_alignment - circle_radius
                vpos = row * 2 * circle_radius + row * per_circle_vpadding + self.voffset + height_center_alignment - circle_radius
                pos = (hpos + circle_radius, vpos + circle_radius)
                self.entries[row][column] = Entry(
                    surface=self.screen,
                    row=row,
                    col=column, 
                    center=pos, 
                    radius=circle_radius, 
                    empty_color=self.empty_colors, 
                    color1=self.color1, 
                    color2=self.color2,
                    state=self.state[row][column], 
                    width=self.width
                )

    def draw(self):
        self.clear_screen()
        rows, columns = self.state.shape
        for i in range(rows):
            for j in range(columns):
                self.entries[i][j].draw()
                
