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

import typing
from survival_lookahead import (
    move_survival_lookahead,
    flood_fill,
    get_opponent_head_positions,
    get_possible_opponent_moves
)

# Constants
X = 'x'
Y = 'y'
LEFT = 'left'
RIGHT = 'right'
DOWN = 'down'
UP = 'up'

# Strategy thresholds
CRITICAL_HEALTH_THRESHOLD = 20
LOW_HEALTH_THRESHOLD = 40
AGGRESSIVE_LENGTH_ADVANTAGE = 2  # Be aggressive when this many segments longer


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "shai-hulud",
        "color": "#E07020",  # Orange like the sandworm
        "head": "sand-worm",
        "tail": "sharp",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


def get_next_position(head: typing.Dict, direction: str) -> typing.Dict:
    """Calculate the next position given a head position and direction."""
    x, y = head[X], head[Y]
    if direction == UP:
        return {X: x, Y: y + 1}
    elif direction == DOWN:
        return {X: x, Y: y - 1}
    elif direction == LEFT:
        return {X: x - 1, Y: y}
    elif direction == RIGHT:
        return {X: x + 1, Y: y}
    return head


def manhattan_distance(pos1: typing.Dict, pos2: typing.Dict) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[X] - pos2[X]) + abs(pos1[Y] - pos2[Y])


def is_head_to_head_risky(pos: typing.Dict, my_length: int, opponents: typing.List, game_state: typing.Dict) -> bool:
    """
    Check if a position is risky due to nearby opponent heads.
    Returns True if an opponent of equal or greater length could move to the same position.
    """
    for opponent in opponents:
        opponent_head = opponent['head']
        opponent_length = opponent['length']

        # Only worry about snakes that are equal or larger
        if opponent_length >= my_length:
            # Check if opponent could move to our target position
            possible_opponent_moves = get_possible_opponent_moves(opponent_head, game_state)
            for opp_move in possible_opponent_moves:
                if opp_move[X] == pos[X] and opp_move[Y] == pos[Y]:
                    return True
    return False


def evaluate_space(pos: typing.Dict, game_state: typing.Dict, my_length: int) -> typing.Tuple[int, int]:
    """
    Evaluate available space using flood fill.
    Returns (space_score, available_space).
    """
    available_space = flood_fill(pos, game_state, max_depth=150)

    # Base score from available space
    space_score = available_space * 10

    # Heavy penalty if space is less than our body length (potential trap)
    if available_space < my_length:
        space_score -= (my_length - available_space) * 100

    return space_score, available_space


def evaluate_food_seeking(pos: typing.Dict, foods: typing.List, my_health: int,
                          is_critical: bool, need_food: bool) -> float:
    """Evaluate how good a position is for getting food."""
    if not foods:
        return 0

    # Find nearest food
    min_distance = min(manhattan_distance(pos, food) for food in foods)

    if is_critical:
        # Critical health: strongly prioritize food
        return 500 / (min_distance + 1)
    elif need_food:
        # Low health: moderately prioritize food
        return 200 / (min_distance + 1)
    else:
        # Healthy: slight preference for food
        return 30 / (min_distance + 1)


def evaluate_head_to_head_defense(pos: typing.Dict, my_length: int,
                                   opponents: typing.List, game_state: typing.Dict) -> int:
    """
    Evaluate head-to-head collision risk.
    Returns a negative score (penalty) for risky positions.
    """
    penalty = 0

    for opponent in opponents:
        opponent_head = opponent['head']
        opponent_length = opponent['length']

        # Get all positions opponent could move to
        possible_opponent_moves = get_possible_opponent_moves(opponent_head, game_state)

        for opp_move in possible_opponent_moves:
            if opp_move[X] == pos[X] and opp_move[Y] == pos[Y]:
                # We could collide head-to-head
                if opponent_length >= my_length:
                    # We would lose or tie (both die)
                    penalty -= 400 + (opponent_length - my_length) * 50
                else:
                    # We would win - this is actually good!
                    penalty += 100

    return penalty


def evaluate_aggressive_hunting(pos: typing.Dict, my_length: int, my_health: int,
                                 opponents: typing.List) -> int:
    """
    Evaluate aggressive hunting opportunities.
    Rewards moving toward smaller snakes when we're healthy and larger.
    """
    score = 0

    # Only hunt when healthy
    if my_health < LOW_HEALTH_THRESHOLD:
        return 0

    for opponent in opponents:
        opponent_head = opponent['head']
        opponent_length = opponent['length']

        length_advantage = my_length - opponent_length

        if length_advantage >= AGGRESSIVE_LENGTH_ADVANTAGE:
            # We're significantly larger - hunt them!
            distance = manhattan_distance(pos, opponent_head)
            # Reward getting closer to smaller snakes
            score += (50 * length_advantage) / (distance + 1)
        elif opponent_length > my_length:
            # They're larger - keep some distance
            distance = manhattan_distance(pos, opponent_head)
            if distance <= 2:
                score -= 50 / (distance + 1)

    return score


def evaluate_edge_avoidance(pos: typing.Dict, board_width: int, board_height: int) -> int:
    """
    Penalize positions near edges to avoid getting trapped.
    """
    penalty = 0

    # Penalize being on the edge
    if pos[X] == 0 or pos[X] == board_width - 1:
        penalty -= 15
    if pos[Y] == 0 or pos[Y] == board_height - 1:
        penalty -= 15

    # Extra penalty for corners
    if (pos[X] == 0 or pos[X] == board_width - 1) and \
       (pos[Y] == 0 or pos[Y] == board_height - 1):
        penalty -= 20

    return penalty


def evaluate_center_preference(pos: typing.Dict, board_width: int, board_height: int) -> int:
    """
    Slight preference for center-board positions.
    More escape routes available from the center.
    """
    center_x = board_width / 2
    center_y = board_height / 2

    distance_to_center = abs(pos[X] - center_x) + abs(pos[Y] - center_y)
    max_distance = center_x + center_y

    # Small bonus for being closer to center
    return 10 * (1 - distance_to_center / max_distance)


def get_strategy_mode(my_health: int, my_length: int, opponents: typing.List) -> str:
    """Determine the current strategy mode based on game state."""
    if my_health < CRITICAL_HEALTH_THRESHOLD:
        return "CRITICAL"

    if my_health < LOW_HEALTH_THRESHOLD:
        return "SURVIVAL"

    # Check if we should be aggressive
    if opponents:
        max_opponent_length = max(opp['length'] for opp in opponents)
        if my_length >= max_opponent_length + AGGRESSIVE_LENGTH_ADVANTAGE:
            return "AGGRESSIVE"

    return "NORMAL"


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    # Get basic game info
    my_head = game_state["you"]["body"][0]
    my_body = game_state["you"]["body"]
    my_length = len(my_body)
    my_health = game_state["you"]["health"]
    my_id = game_state["you"]["id"]

    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    foods = game_state['board']['food']

    # Get opponents
    opponents = get_opponent_head_positions(game_state, my_id)

    # Determine strategy mode
    mode = get_strategy_mode(my_health, my_length, opponents)
    is_critical = my_health < CRITICAL_HEALTH_THRESHOLD
    need_food = my_health < LOW_HEALTH_THRESHOLD

    # Get safe moves using lookahead
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

    # Score each safe move
    move_scores = {}
    move_spaces = {}

    for direction in safe_moves:
        next_pos = get_next_position(my_head, direction)
        score = 0

        # 1. Space evaluation (most important for survival)
        space_score, available_space = evaluate_space(next_pos, game_state, my_length)
        score += space_score
        move_spaces[direction] = available_space

        # 2. Food seeking (importance varies by health)
        if foods:
            score += evaluate_food_seeking(next_pos, foods, my_health, is_critical, need_food)

        # 3. Head-to-head defense
        if opponents:
            score += evaluate_head_to_head_defense(next_pos, my_length, opponents, game_state)

        # 4. Aggressive hunting (when we're bigger)
        if mode == "AGGRESSIVE" and opponents:
            score += evaluate_aggressive_hunting(next_pos, my_length, my_health, opponents)

        # 5. Edge avoidance
        score += evaluate_edge_avoidance(next_pos, board_width, board_height)

        # 6. Center preference (slight)
        score += evaluate_center_preference(next_pos, board_width, board_height)

        move_scores[direction] = score

    # Choose the best move
    next_move = max(safe_moves, key=lambda m: move_scores[m])

    print(f"MOVE {game_state['turn']}: {next_move} | Mode: {mode} | "
          f"Score: {move_scores[next_move]:.0f} | Space: {move_spaces[next_move]} | "
          f"Health: {my_health}")

    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
