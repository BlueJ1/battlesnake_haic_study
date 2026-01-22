import typing
from copy import deepcopy
import numpy as np

X = 'x'
Y = 'y'
LEFT = 'left'
RIGHT = 'right'
DOWN = 'down'
UP = 'up'


def move_survival_lookahead(game_state, lookahead_depth):
    if lookahead_depth == 0:
        return True
    # elif lookahead_depth < 0:
    #     print('Something went very wrong')
    #     print(lookahead_depth)
    #     print(game_state)
    #     exit(0)

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # Step 1 - Prevent my Battlesnake from moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    if my_head["y"] == 0:
        is_move_safe["down"] = False
    if my_head["y"] == board_height - 1:
        is_move_safe["up"] = False
    if my_head["x"] == board_width - 1:
        is_move_safe["right"] = False
    if my_head["x"] == 0:
        is_move_safe["left"] = False

    # Step 2 - Prevent my Battlesnake from colliding with itself
    my_body = game_state['you']['body']
    my_x = my_head[X]
    my_y = my_head[Y]
    coordinates_to_direction = {coordinates: direction for direction, coordinates in
                                {LEFT: (my_x - 1, my_y), RIGHT: (my_x + 1, my_y), UP: (my_x, my_y + 1),
                                 DOWN: (my_x, my_y - 1)}.items() if is_move_safe[direction]}
    for body_part in my_body:
        if (body_part[X], body_part[Y]) in coordinates_to_direction.keys():
            is_move_safe[coordinates_to_direction[(body_part[X], body_part[Y])]] = False

    # Step 3 - Prevent my Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']
    coordinates_to_direction = {coordinates: direction for coordinates, direction in coordinates_to_direction.items() if
                                is_move_safe[direction]}
    for opponent in opponents:
        if opponent["body"][0] == my_head:
            continue
        for body_part in opponent['body']:
            if (body_part[X], body_part[Y]) in coordinates_to_direction.keys():
                is_move_safe[coordinates_to_direction[(body_part[X], body_part[Y])]] = False

    # Are there any safe moves left?
    currently_safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            currently_safe_moves.append(move)

    if lookahead_depth == 1:
        return currently_safe_moves

    safe_moves = []
    # Step 4 - Check deeper paths
    for currently_safe_move in currently_safe_moves:
        # TODO: extra loop for updating game state with possible opponent movements
        new_game_state = update_game_state(game_state, currently_safe_move)
        next_safe_moves = move_survival_lookahead(new_game_state, lookahead_depth=lookahead_depth - 1)
        if next_safe_moves:
            safe_moves.append(currently_safe_move)

    return safe_moves


def update_game_state(game_state: typing.Dict, move):
    game_state = game_state.copy()
    game_state['board'] = deepcopy(game_state['board'])
    game_state['you'] = deepcopy(game_state['you'])
    direction = None
    my_head = game_state["you"]["body"][0]
    if move == LEFT:
        direction = [my_head[X] - 1, my_head[Y]]
    elif move == RIGHT:
        direction = [my_head[X] + 1, my_head[Y]]
    elif move == UP:
        direction = [my_head[X], my_head[Y] + 1]
    elif move == DOWN:
        direction = [my_head[X], my_head[Y] - 1]
    # else:
        # print('Something went wrong with updating the game state')
        # print(move)
        # print(game_state)
        # exit(0)

    xy_point = {X: direction[0], Y:direction[1]}
    my_head = game_state["you"]["body"][0]
    my_snake_idx = np.argmax([points_equal(snake['body'][0], my_head) for snake in game_state["board"]["snakes"]])
    game_state["you"]["body"] = [xy_point] + game_state["you"]["body"][:-1] + ([game_state["you"]["body"][-1]] if food_there(game_state['board']['food'], xy_point) else [])
    game_state["board"]["snakes"][my_snake_idx]['body'] = game_state["you"]["body"]
    return game_state


def food_there(foods, point):
    for food in foods:
        if points_equal(food, point):
            return True
    return False


def points_equal(point1, point2):
    return point1[X] == point2[X] and point1[Y] == point2[Y]
