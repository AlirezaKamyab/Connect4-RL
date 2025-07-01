import pygame

class Entry:
    EMPTY, PLAYER1, PLAYER2 = 0, 1, 2
    def __init__(
            self,
            surface: pygame.Surface,
            center: tuple, 
            radius: float,
            row:int,
            col:int,
            empty_color: tuple=(0, 0, 0),
            color1: tuple=(0, 0, 0),
            color2: tuple=(0, 0, 0),
            state:int=EMPTY,
            width:int=2
        ):
        self.surface = surface
        self.center = center
        self.radius = radius
        self.width = width

        # These two variables specify where this entry is on the board
        self.row = row
        self.col = col

        # Empty color: this color is defined for empty state
        self.empty_color = empty_color
        # Color1: this color shows the entry has been taken by the first player
        self.color1 = color1
        # Color2: this color shows the entry has been taken by the second player
        self.color2 = color2
        # This is the current color 
        self.change_state(state)

    def on_entry(self, pos:tuple) -> bool:
        width, height = pos
        center_width, center_height = self.center
        if not(width < center_width + self.radius and width > center_width - self.radius):
            return False
        if not(height < center_height + self.radius and height > center_height - self.radius):
            return False
        return True
    
    def change_state(self, new_state):
        self.state = new_state
        colors = (self.empty_color, self.color1, self.color2)
        self.color = colors[self.state]
    
    def draw(self):
        pygame.draw.circle(
            surface=self.surface, 
            color=self.color, 
            center=self.center, 
            radius=self.radius,
            width=self.width if self.state == Entry.EMPTY else 0
        )