import typing
from copy import deepcopy
from collections import deque
import numpy as np

X = 'x'
Y = 'y'
LEFT = 'left'
RIGHT = 'right'
DOWN = 'down'
UP = 'up'


def flood_fill(start_pos: typing.Dict, game_state: typing.Dict, max_depth: int = None) -> int:
    """
    Use flood fill (BFS) to calculate available space from a starting position.
    Returns the number of reachable squares.
    Dynamically adjusts search depth based on board size.
    """
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    all_snakes = game_state['board']['snakes']
    my_length = len(game_state['you']['body'])

    # Dynamically calculate maximum iterations based on board size
    board_size = board_width * board_height
    if max_depth is None:
        if board_size <= 100:  # Small map (7x7, 10x10)
            max_depth = 80
        elif board_size <= 200:  # Medium map (11x11, 13x13)
            max_depth = 120
        else:  # Large map (19x19, 25x25)
            max_depth = min(board_size // 2, 250)

    # Safe space threshold for early termination
    safe_space = my_length * 3

    # Build obstacle set (all snake bodies except tails which will move)
    obstacles = set()
    for snake in all_snakes:
        body = snake['body'][:-1] if len(snake['body']) > 1 else snake['body']
        for segment in body:
            obstacles.add((segment[X], segment[Y]))

    visited = set()
    queue = deque([start_pos])
    visited.add((start_pos[X], start_pos[Y]))
    count = 0

    while queue and count < max_depth:
        pos = queue.popleft()
        count += 1

        # Early termination optimization - if we found enough space, stop
        if count >= safe_space:
            break

        # Check all four directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x = pos[X] + dx
            next_y = pos[Y] + dy
            pos_tuple = (next_x, next_y)

            # Skip if already visited
            if pos_tuple in visited:
                continue

            # Skip if out of bounds
            if next_x < 0 or next_x >= board_width or next_y < 0 or next_y >= board_height:
                continue

            # Skip if occupied by snake
            if pos_tuple in obstacles:
                continue

            visited.add(pos_tuple)
            queue.append({X: next_x, Y: next_y})

    return count


def get_opponent_head_positions(game_state: typing.Dict, my_id: str) -> typing.List[typing.Dict]:
    """Get list of opponent head positions."""
    opponents = []
    for snake in game_state['board']['snakes']:
        if snake['id'] != my_id:
            opponents.append({
                'head': snake['body'][0],
                'length': len(snake['body']),
                'id': snake['id']
            })
    return opponents


def get_possible_opponent_moves(opponent_head: typing.Dict, game_state: typing.Dict) -> typing.List[typing.Dict]:
    """Get all possible next positions for an opponent head."""
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    possible = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        next_x = opponent_head[X] + dx
        next_y = opponent_head[Y] + dy

        if 0 <= next_x < board_width and 0 <= next_y < board_height:
            possible.append({X: next_x, Y: next_y})

    return possible


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
    # Exclude tail unless we're about to grow (tail == second-to-last means growth)
    body_to_check = my_body
    if len(body_to_check) > 1:
        tail = body_to_check[-1]
        second_last = body_to_check[-2]
        if tail[X] != second_last[X] or tail[Y] != second_last[Y]:
            body_to_check = body_to_check[:-1]  # Exclude tail since it will move
    for body_part in body_to_check:
        if (body_part[X], body_part[Y]) in coordinates_to_direction.keys():
            is_move_safe[coordinates_to_direction[(body_part[X], body_part[Y])]] = False

    # Step 3 - Prevent my Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']
    coordinates_to_direction = {coordinates: direction for coordinates, direction in coordinates_to_direction.items() if
                                is_move_safe[direction]}
    for opponent in opponents:
        if opponent["body"][0] == my_head:
            continue
        # Exclude tail unless snake might have just eaten (tail == second-to-last segment means growth)
        body_to_check = opponent['body']
        if len(body_to_check) > 1:
            # If tail position != second-to-last position, tail will move away
            tail = body_to_check[-1]
            second_last = body_to_check[-2]
            if tail[X] != second_last[X] or tail[Y] != second_last[Y]:
                body_to_check = body_to_check[:-1]  # Exclude tail
        for body_part in body_to_check:
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
        # Don't simulate opponent moves in lookahead to be more conservative
        # The opponent might not move where we predict
        new_game_state = update_game_state(game_state, currently_safe_move, simulate_opponents=False)
        next_safe_moves = move_survival_lookahead(new_game_state, lookahead_depth=lookahead_depth - 1)
        if next_safe_moves:
            safe_moves.append(currently_safe_move)

    return safe_moves


def get_best_opponent_move(opponent: typing.Dict, game_state: typing.Dict) -> typing.Dict:
    """
    Predict the most likely move for an opponent snake.
    Uses simple heuristics: avoid walls, avoid collisions, prefer space.
    """
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    opponent_head = opponent['body'][0]
    opponent_body = opponent['body']

    best_move = None
    best_score = -float('inf')

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        next_x = opponent_head[X] + dx
        next_y = opponent_head[Y] + dy

        # Skip if out of bounds
        if next_x < 0 or next_x >= board_width or next_y < 0 or next_y >= board_height:
            continue

        # Skip if collides with own body (excluding tail)
        body_to_check = opponent_body[:-1] if len(opponent_body) > 1 else []
        if any(next_x == seg[X] and next_y == seg[Y] for seg in body_to_check):
            continue

        # Skip if collides with other snakes
        collision = False
        for snake in game_state['board']['snakes']:
            if snake['id'] == opponent['id']:
                continue
            snake_body = snake['body'][:-1] if len(snake['body']) > 1 else snake['body']
            if any(next_x == seg[X] and next_y == seg[Y] for seg in snake_body):
                collision = True
                break
        if collision:
            continue

        # Score based on available space (simplified)
        score = 0
        # Prefer center positions
        center_x = board_width / 2
        center_y = board_height / 2
        score -= abs(next_x - center_x) + abs(next_y - center_y)

        # Avoid edges
        if next_x == 0 or next_x == board_width - 1:
            score -= 5
        if next_y == 0 or next_y == board_height - 1:
            score -= 5

        if score > best_score:
            best_score = score
            best_move = {X: next_x, Y: next_y}

    # If no valid move found, return current head position (snake will die anyway)
    return best_move if best_move else opponent_head


def update_game_state(game_state: typing.Dict, move, simulate_opponents: bool = True):
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

    xy_point = {X: direction[0], Y:direction[1]}
    my_head = game_state["you"]["body"][0]
    my_snake_idx = np.argmax([points_equal(snake['body'][0], my_head) for snake in game_state["board"]["snakes"]])
    game_state["you"]["body"] = [xy_point] + game_state["you"]["body"][:-1] + ([game_state["you"]["body"][-1]] if food_there(game_state['board']['food'], xy_point) else [])
    game_state["board"]["snakes"][my_snake_idx]['body'] = game_state["you"]["body"]

    # Simulate opponent movements
    if simulate_opponents:
        my_id = game_state['you']['id']
        for i, snake in enumerate(game_state['board']['snakes']):
            if snake['id'] == my_id:
                continue

            # Get predicted next position for opponent
            next_pos = get_best_opponent_move(snake, game_state)

            # Update opponent's body (move forward, remove tail unless eating)
            ate_food = food_there(game_state['board']['food'], next_pos)
            new_body = [next_pos] + snake['body'][:-1]
            if ate_food:
                new_body.append(snake['body'][-1])
            game_state['board']['snakes'][i]['body'] = new_body

    return game_state


def food_there(foods, point):
    for food in foods:
        if points_equal(food, point):
            return True
    return False


def points_equal(point1, point2):
    return point1[X] == point2[X] and point1[Y] == point2[Y]
