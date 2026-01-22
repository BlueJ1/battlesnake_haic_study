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
from minimax import get_minimax_move, MinimaxSearch

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
AGGRESSIVE_LENGTH_ADVANTAGE = 2  # Default, will be dynamically adjusted
HUNT_REWARD_THRESHOLD = 3  # Minimum length to gain from hunting


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


def calculate_dynamic_aggressive_threshold(game_state: typing.Dict) -> int:
    """
    Dynamically calculate aggressive threshold based on current game state.
    Adapts strategy based on opponent count and relative lengths.
    """
    my_length = len(game_state['you']['body'])
    opponents = [s for s in game_state['board']['snakes']
                 if s['id'] != game_state['you']['id']]

    if not opponents:
        return 1  # No opponents, can be aggressive immediately

    # Calculate average and max opponent lengths
    avg_opponent_length = sum(len(s['body']) for s in opponents) / len(opponents)
    max_opponent_length = max(len(s['body']) for s in opponents)

    # If opponents are generally much stronger, be conservative
    if avg_opponent_length > my_length + 3:
        return 4

    # If strongest opponent is much stronger, be conservative
    if max_opponent_length > my_length + 4:
        return 4

    # If opponents are generally weaker, be aggressive
    if avg_opponent_length < my_length - 2:
        return 1

    # Many opponents, be conservative
    if len(opponents) >= 4:
        return 3

    # Few opponents, can be somewhat aggressive
    if len(opponents) <= 2:
        return 2

    # Default
    return 2


def predict_opponent_moves(opponent: typing.Dict, game_state: typing.Dict, depth: int = 2) -> typing.List[typing.Dict]:
    """
    Predict opponent's next positions up to 'depth' moves ahead.
    Returns list of all possible positions the opponent could reach.
    """
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    opponent_head = opponent['head']
    opponent_body = [{'x': seg['x'], 'y': seg['y']} for seg in opponent.get('body', [opponent_head])]

    predicted_positions = []
    first_step_positions = []

    # First step predictions
    for direction in [UP, DOWN, LEFT, RIGHT]:
        pos1 = get_next_position(opponent_head, direction)

        # Skip if out of bounds
        if pos1[X] < 0 or pos1[X] >= board_width or pos1[Y] < 0 or pos1[Y] >= board_height:
            continue

        # Skip if collides with own body (excluding tail)
        body_to_check = opponent_body[:-1] if len(opponent_body) > 1 else []
        if any(pos1[X] == seg['x'] and pos1[Y] == seg['y'] for seg in body_to_check):
            continue

        predicted_positions.append(pos1)
        first_step_positions.append(pos1)

    # Second step predictions
    if depth >= 2:
        for pos1 in first_step_positions:
            for direction in [UP, DOWN, LEFT, RIGHT]:
                pos2 = get_next_position(pos1, direction)

                # Skip if out of bounds
                if pos2[X] < 0 or pos2[X] >= board_width or pos2[Y] < 0 or pos2[Y] >= board_height:
                    continue

                # Check collision with future body state
                future_body = [pos1] + opponent_body[:-2] if len(opponent_body) > 2 else [pos1]
                if any(pos2[X] == seg['x'] and pos2[Y] == seg['y'] for seg in future_body):
                    continue

                predicted_positions.append(pos2)

    return predicted_positions


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
        space_score -= (my_length - available_space) * 150  # Increased penalty
    # Moderate penalty if space is less than 2x body length (risky)
    elif available_space < my_length * 2:
        space_score -= (my_length * 2 - available_space) * 20
    # Small bonus for having lots of space
    elif available_space > my_length * 3:
        space_score += 50

    return space_score, available_space


def evaluate_food_seeking(pos: typing.Dict, foods: typing.List, my_health: int,
                          is_critical: bool, need_food: bool,
                          my_head: typing.Dict, my_length: int,
                          opponents: typing.List, game_state: typing.Dict) -> float:
    """
    Smart food selection - evaluates food based on distance, threats, and space after eating.
    """
    if not foods:
        return 0

    # Evaluate each food option
    food_evaluations = []

    for food in foods:
        # Distance score - closer is better
        distance = manhattan_distance(pos, food)
        distance_score = 100.0 / (distance + 1)

        # Threat assessment - penalize food that opponents are closer to
        threat_score = 0
        my_distance_to_food = manhattan_distance(my_head, food)

        for opponent in opponents:
            opp_distance = manhattan_distance(opponent['head'], food)

            # If opponent is closer or equal distance and larger/equal, it's risky
            if opp_distance <= my_distance_to_food and opponent['length'] >= my_length:
                threat_score -= 50.0 / (opp_distance + 1)
            # If we're closer and larger, bonus for stealing food
            elif opp_distance > my_distance_to_food and opponent['length'] < my_length:
                threat_score += 20.0 / (my_distance_to_food + 1)

        # Space estimation after reaching food (simplified)
        # Penalize food near edges/corners
        space_penalty = 0
        board_width = game_state['board']['width']
        board_height = game_state['board']['height']
        if food[X] == 0 or food[X] == board_width - 1:
            space_penalty -= 10
        if food[Y] == 0 or food[Y] == board_height - 1:
            space_penalty -= 10
        if (food[X] == 0 or food[X] == board_width - 1) and \
           (food[Y] == 0 or food[Y] == board_height - 1):
            space_penalty -= 15  # Corner penalty

        total_score = distance_score + threat_score + space_penalty
        food_evaluations.append((food, total_score))

    # Find the best food
    best_food, _ = max(food_evaluations, key=lambda x: x[1])

    # Score based on moving toward best food
    distance_to_best = manhattan_distance(pos, best_food)

    if is_critical:
        # Critical health: strongly prioritize food
        return 500 / (distance_to_best + 1)
    elif need_food:
        # Low health: moderately prioritize food
        return 200 / (distance_to_best + 1)
    else:
        # Healthy: slight preference for food
        return 30 / (distance_to_best + 1)


def evaluate_head_to_head_defense(pos: typing.Dict, my_length: int,
                                   opponents: typing.List, game_state: typing.Dict) -> int:
    """
    Evaluate head-to-head collision risk using 2-step prediction.
    Returns a negative score (penalty) for risky positions.
    """
    penalty = 0

    for opponent in opponents:
        opponent_head = opponent['head']
        opponent_length = opponent['length']

        # Use 2-step prediction for larger/equal opponents
        if opponent_length >= my_length:
            # Build opponent dict for prediction
            opponent_snake = None
            for snake in game_state['board']['snakes']:
                if snake['id'] == opponent['id']:
                    opponent_snake = {'head': snake['body'][0], 'body': snake['body']}
                    break

            if opponent_snake:
                predicted_positions = predict_opponent_moves(opponent_snake, game_state, depth=2)

                for pred_pos in predicted_positions:
                    if pred_pos[X] == pos[X] and pred_pos[Y] == pos[Y]:
                        distance = manhattan_distance(pos, opponent_head)

                        if distance <= 1:
                            # Immediate threat
                            penalty -= 400 + (opponent_length - my_length) * 50
                        else:
                            # 2-step threat (less severe)
                            penalty -= 150 + (opponent_length - my_length) * 30
                        break
        else:
            # Smaller opponent - check if we can hunt them
            possible_opponent_moves = get_possible_opponent_moves(opponent_head, game_state)
            for opp_move in possible_opponent_moves:
                if opp_move[X] == pos[X] and opp_move[Y] == pos[Y]:
                    # We would win this collision!
                    penalty += 100
                    break

    return penalty


def evaluate_aggressive_hunting(pos: typing.Dict, my_length: int, my_health: int,
                                 opponents: typing.List, dynamic_threshold: int) -> int:
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

        if length_advantage >= dynamic_threshold:
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


def evaluate_tail_chase(pos: typing.Dict, my_body: typing.List, available_space: int,
                        my_length: int) -> int:
    """
    Evaluate tail-chasing as a safe strategy.
    Moving toward your own tail guarantees an escape route since tail moves away.
    Only activates when space is limited.
    """
    if len(my_body) < 3:
        return 0

    my_tail = my_body[-1]
    distance_to_tail = abs(pos[X] - my_tail[X]) + abs(pos[Y] - my_tail[Y])

    # When space is very limited, prioritize staying close to tail
    if available_space < my_length:
        # Critical: chase tail aggressively
        return 60 / (distance_to_tail + 1)
    elif available_space < my_length * 2:
        # Low space: moderate tail chase
        return 25 / (distance_to_tail + 1)

    return 0  # No tail chasing when space is plentiful


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

    # Calculate dynamic aggressive threshold
    dynamic_threshold = calculate_dynamic_aggressive_threshold(game_state)

    # Determine strategy mode
    mode = get_strategy_mode(my_health, my_length, opponents)
    is_critical = my_health < CRITICAL_HEALTH_THRESHOLD
    need_food = my_health < LOW_HEALTH_THRESHOLD

    # Get safe moves using lookahead (for fallback)
    lookahead_depth = 4
    safe_moves = []
    while lookahead_depth > 0:
        safe_moves = move_survival_lookahead(game_state, lookahead_depth=lookahead_depth)
        if len(safe_moves) > 0:
            break
        else:
            lookahead_depth -= 1

    if len(safe_moves) == 0:
        # Emergency: no safe moves, pick least bad option
        print(f"MOVE {game_state['turn']}: No safe moves detected! Picking emergency move")
        # Try to at least stay in bounds
        emergency_moves = []
        if my_head[Y] > 0:
            emergency_moves.append("down")
        if my_head[Y] < board_height - 1:
            emergency_moves.append("up")
        if my_head[X] > 0:
            emergency_moves.append("left")
        if my_head[X] < board_width - 1:
            emergency_moves.append("right")
        if emergency_moves:
            return {"move": emergency_moves[0]}
        return {"move": "down"}  # Last resort

    # Use minimax with alpha-beta pruning for move selection
    # Time limit: 300ms to stay well under the 500ms soft limit
    minimax_move, minimax_score, depth_reached = get_minimax_move(game_state, time_limit_ms=300)

    # Validate minimax result against safe moves
    if minimax_move and minimax_move in safe_moves:
        next_move = minimax_move
        print(f"MOVE {game_state['turn']}: {next_move} | Mode: {mode} | "
              f"Minimax Score: {minimax_score:.0f} | Depth: {depth_reached} | "
              f"Health: {my_health}")
        return {"move": next_move}

    # Fallback to heuristic-based scoring if minimax fails or returns unsafe move
    print(f"MOVE {game_state['turn']}: Minimax returned {minimax_move}, falling back to heuristics")

    # Score each safe move using heuristics
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
            score += evaluate_food_seeking(next_pos, foods, my_health, is_critical, need_food,
                                          my_head, my_length, opponents, game_state)

        # 3. Head-to-head defense
        if opponents:
            score += evaluate_head_to_head_defense(next_pos, my_length, opponents, game_state)

        # 4. Aggressive hunting (when we're bigger)
        if mode == "AGGRESSIVE" and opponents:
            score += evaluate_aggressive_hunting(next_pos, my_length, my_health, opponents, dynamic_threshold)

        # 5. Edge avoidance
        score += evaluate_edge_avoidance(next_pos, board_width, board_height)

        # 6. Center preference (slight)
        score += evaluate_center_preference(next_pos, board_width, board_height)

        # 7. Tail chasing (minimal improvement for survival)
        score += evaluate_tail_chase(next_pos, my_body, available_space, my_length)

        move_scores[direction] = score

    # Choose the best move
    next_move = max(safe_moves, key=lambda m: move_scores[m])

    print(f"MOVE {game_state['turn']}: {next_move} | Mode: {mode} | "
          f"Heuristic Score: {move_scores[next_move]:.0f} | Space: {move_spaces[next_move]} | "
          f"Health: {my_health}")

    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
