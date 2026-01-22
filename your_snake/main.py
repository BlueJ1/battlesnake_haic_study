# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
import numpy as np
from survival_lookahead import move_survival_lookahead


X = 'x'
Y = 'y'
LEFT = 'left'
RIGHT = 'right'
DOWN = 'down'
UP = 'up'


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "shai-hulud",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    print(game_state.keys())

    lookahead_depth = 4
    safe_moves = []
    while lookahead_depth > 0:
        safe_moves = move_survival_lookahead(game_state, lookahead_depth=lookahead_depth)
        if len(safe_moves) > 0:
            break
        else:
            lookahead_depth -= 1
    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}
    foods = game_state['board']['food']
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_x = my_head[X]
    my_y = my_head[Y]
    safe_moves_coordinates_to_directions = {coordinates: direction for direction, coordinates in
                                            {LEFT: (my_x - 1, my_y), RIGHT: (my_x + 1, my_y), UP: (my_x, my_y + 1),
                                             DOWN: (my_x, my_y - 1)}.items() if direction in safe_moves}
    min_distance_from_food = np.array(
        [[direction, min([abs(food[X] - coordinates[0]) + abs(food[Y] - coordinates[1]) for food in foods])] for
         coordinates, direction in safe_moves_coordinates_to_directions.items()])
    next_move = min_distance_from_food[np.argmin(min_distance_from_food[:, 1])][0]
    print(f"MOVE {game_state['turn']}: {next_move}")

    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
