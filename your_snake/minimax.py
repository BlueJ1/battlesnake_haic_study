"""
Minimax with Alpha-Beta Pruning for Battlesnake

This module implements adversarial search with:
- Alpha-beta pruning for efficient tree search
- Time-sensitive iterative deepening
- Move ordering for better pruning
"""

import typing
import time
from collections import deque

# Constants
X = 'x'
Y = 'y'
LEFT = 'left'
RIGHT = 'right'
DOWN = 'down'
UP = 'up'

DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Evaluation weights
SPACE_WEIGHT = 10.0
FOOD_WEIGHT = 5.0
HEAD_TO_HEAD_WEIGHT = 15.0
CENTER_WEIGHT = 1.0
EDGE_PENALTY = 3.0
LENGTH_WEIGHT = 20.0


class MinimaxSearch:
    """
    Minimax search with alpha-beta pruning and iterative deepening.
    """

    def __init__(self, time_limit_ms: float = 300):
        """
        Initialize minimax search.

        Args:
            time_limit_ms: Time limit for search in milliseconds
        """
        self.time_limit = time_limit_ms / 1000.0  # Convert to seconds
        self.start_time = 0
        self.nodes_evaluated = 0
        self.best_move_at_depth = {}
        self.search_aborted = False

    def is_time_up(self) -> bool:
        """Check if we've exceeded time limit."""
        return time.time() - self.start_time >= self.time_limit

    def get_legal_moves(self, game_state: typing.Dict, snake_id: str) -> typing.List[str]:
        """
        Get all legal moves for a snake.

        Args:
            game_state: Current game state
            snake_id: ID of the snake to get moves for

        Returns:
            List of legal move directions
        """
        # Find the snake
        snake = None
        for s in game_state['board']['snakes']:
            if s['id'] == snake_id:
                snake = s
                break

        if not snake or len(snake['body']) == 0:
            return []

        head = snake['body'][0]
        body = snake['body']
        board_width = game_state['board']['width']
        board_height = game_state['board']['height']

        legal_moves = []

        for direction in DIRECTIONS:
            next_pos = self._get_next_position(head, direction)

            # Check bounds
            if next_pos[X] < 0 or next_pos[X] >= board_width:
                continue
            if next_pos[Y] < 0 or next_pos[Y] >= board_height:
                continue

            # Check self-collision (excluding tail which will move)
            body_to_check = body[:-1] if len(body) > 1 else []
            # If tail == second-to-last, snake is growing, include tail
            if len(body) > 1 and body[-1][X] == body[-2][X] and body[-1][Y] == body[-2][Y]:
                body_to_check = body

            collision = False
            for seg in body_to_check:
                if next_pos[X] == seg[X] and next_pos[Y] == seg[Y]:
                    collision = True
                    break
            if collision:
                continue

            # Check collision with other snakes' bodies (excluding their tails)
            for other_snake in game_state['board']['snakes']:
                if other_snake['id'] == snake_id:
                    continue
                other_body = other_snake['body']
                other_body_to_check = other_body[:-1] if len(other_body) > 1 else other_body
                # If other snake is growing, include tail
                if len(other_body) > 1 and other_body[-1][X] == other_body[-2][X] and other_body[-1][Y] == other_body[-2][Y]:
                    other_body_to_check = other_body

                for seg in other_body_to_check:
                    if next_pos[X] == seg[X] and next_pos[Y] == seg[Y]:
                        collision = True
                        break
                if collision:
                    break

            if not collision:
                legal_moves.append(direction)

        return legal_moves

    def _get_next_position(self, head: typing.Dict, direction: str) -> typing.Dict:
        """Calculate next position given head and direction."""
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

    def apply_move(self, game_state: typing.Dict, snake_id: str, move: str) -> typing.Dict:
        """
        Apply a move to the game state and return new state.

        Args:
            game_state: Current game state
            snake_id: ID of the snake making the move
            move: Direction to move

        Returns:
            New game state after the move
        """
        # Deep copy only the parts we need to modify
        new_state = {
            'board': {
                'width': game_state['board']['width'],
                'height': game_state['board']['height'],
                'food': list(game_state['board']['food']),
                'snakes': []
            },
            'you': None
        }

        for snake in game_state['board']['snakes']:
            if snake['id'] == snake_id:
                head = snake['body'][0]
                new_head = self._get_next_position(head, move)

                # Check if eating food
                ate_food = any(
                    food[X] == new_head[X] and food[Y] == new_head[Y]
                    for food in game_state['board']['food']
                )

                # Build new body
                if ate_food:
                    new_body = [new_head] + list(snake['body'])
                    # Remove food from board
                    new_state['board']['food'] = [
                        f for f in new_state['board']['food']
                        if not (f[X] == new_head[X] and f[Y] == new_head[Y])
                    ]
                else:
                    new_body = [new_head] + list(snake['body'][:-1])

                new_snake = {
                    'id': snake['id'],
                    'body': new_body,
                    'health': snake['health'] - 1 if not ate_food else 100
                }
                new_state['board']['snakes'].append(new_snake)

                if snake['id'] == game_state['you']['id']:
                    new_state['you'] = new_snake
            else:
                # Copy opponent snake unchanged for now
                new_state['board']['snakes'].append({
                    'id': snake['id'],
                    'body': list(snake['body']),
                    'health': snake['health']
                })

        if new_state['you'] is None:
            new_state['you'] = game_state['you']

        return new_state

    def apply_opponent_moves(self, game_state: typing.Dict, my_id: str) -> typing.Dict:
        """
        Apply predicted opponent moves (opponents move toward center/away from us).

        Args:
            game_state: Current game state
            my_id: Our snake's ID

        Returns:
            New game state with opponent moves applied
        """
        new_state = {
            'board': {
                'width': game_state['board']['width'],
                'height': game_state['board']['height'],
                'food': list(game_state['board']['food']),
                'snakes': []
            },
            'you': None
        }

        for snake in game_state['board']['snakes']:
            if snake['id'] == my_id:
                new_state['board']['snakes'].append({
                    'id': snake['id'],
                    'body': list(snake['body']),
                    'health': snake['health']
                })
                new_state['you'] = new_state['board']['snakes'][-1]
            else:
                # Get opponent's best move (simple heuristic)
                legal_moves = self.get_legal_moves(game_state, snake['id'])
                if not legal_moves:
                    # Opponent has no moves - they die, skip them
                    continue

                # Pick move that maximizes space (simplified)
                best_move = legal_moves[0]
                best_score = -float('inf')

                for move in legal_moves:
                    next_pos = self._get_next_position(snake['body'][0], move)
                    score = self._quick_position_score(next_pos, game_state)
                    if score > best_score:
                        best_score = score
                        best_move = move

                # Apply move
                head = snake['body'][0]
                new_head = self._get_next_position(head, best_move)
                ate_food = any(
                    food[X] == new_head[X] and food[Y] == new_head[Y]
                    for food in game_state['board']['food']
                )

                if ate_food:
                    new_body = [new_head] + list(snake['body'])
                    new_state['board']['food'] = [
                        f for f in new_state['board']['food']
                        if not (f[X] == new_head[X] and f[Y] == new_head[Y])
                    ]
                else:
                    new_body = [new_head] + list(snake['body'][:-1])

                new_state['board']['snakes'].append({
                    'id': snake['id'],
                    'body': new_body,
                    'health': snake['health'] - 1 if not ate_food else 100
                })

        if new_state['you'] is None:
            new_state['you'] = game_state['you']

        return new_state

    def _quick_position_score(self, pos: typing.Dict, game_state: typing.Dict) -> float:
        """Quick heuristic score for a position (used for opponent move prediction)."""
        board_width = game_state['board']['width']
        board_height = game_state['board']['height']
        center_x = board_width / 2
        center_y = board_height / 2

        # Prefer center
        score = -abs(pos[X] - center_x) - abs(pos[Y] - center_y)

        # Avoid edges
        if pos[X] == 0 or pos[X] == board_width - 1:
            score -= 2
        if pos[Y] == 0 or pos[Y] == board_height - 1:
            score -= 2

        return score

    def is_terminal(self, game_state: typing.Dict) -> typing.Tuple[bool, float]:
        """
        Check if game state is terminal (game over for us).

        Returns:
            (is_terminal, score) - score is very negative if we died, very positive if we won
        """
        my_id = game_state['you']['id']
        my_snake = None
        opponents_alive = 0

        for snake in game_state['board']['snakes']:
            if snake['id'] == my_id:
                my_snake = snake
            else:
                opponents_alive += 1

        # We died
        if my_snake is None or len(my_snake['body']) == 0:
            return True, -10000

        # We won (all opponents dead)
        if opponents_alive == 0:
            return True, 10000

        # Check if we have no legal moves (we will die)
        legal_moves = self.get_legal_moves(game_state, my_id)
        if not legal_moves:
            return True, -10000

        # Check starvation
        if my_snake['health'] <= 0:
            return True, -10000

        return False, 0

    def evaluate(self, game_state: typing.Dict) -> float:
        """
        Evaluate the game state from our perspective.
        Higher scores are better for us.

        Args:
            game_state: Current game state

        Returns:
            Evaluation score
        """
        my_id = game_state['you']['id']
        my_snake = None
        opponents = []

        for snake in game_state['board']['snakes']:
            if snake['id'] == my_id:
                my_snake = snake
            else:
                opponents.append(snake)

        if my_snake is None:
            return -10000

        my_head = my_snake['body'][0]
        my_length = len(my_snake['body'])
        my_health = my_snake['health']
        board_width = game_state['board']['width']
        board_height = game_state['board']['height']

        score = 0.0

        # 1. Space control (most important)
        available_space = self._flood_fill_fast(my_head, game_state)
        if available_space < my_length:
            # Trapped - very bad
            score -= (my_length - available_space) * 100
        else:
            score += available_space * SPACE_WEIGHT

        # 2. Length advantage
        if opponents:
            max_opp_length = max(len(opp['body']) for opp in opponents)
            length_diff = my_length - max_opp_length
            score += length_diff * LENGTH_WEIGHT

        # 3. Health/Food proximity
        if game_state['board']['food']:
            min_food_dist = min(
                abs(my_head[X] - food[X]) + abs(my_head[Y] - food[Y])
                for food in game_state['board']['food']
            )
            # More important when health is low
            health_urgency = (100 - my_health) / 100.0
            score += (10 - min_food_dist) * FOOD_WEIGHT * (1 + health_urgency)

        # 4. Head-to-head threat assessment
        for opp in opponents:
            opp_head = opp['body'][0]
            opp_length = len(opp['body'])
            dist_to_opp = abs(my_head[X] - opp_head[X]) + abs(my_head[Y] - opp_head[Y])

            if dist_to_opp <= 2:
                if opp_length >= my_length:
                    # Dangerous - they could kill us
                    score -= HEAD_TO_HEAD_WEIGHT * (3 - dist_to_opp)
                else:
                    # We could kill them
                    score += HEAD_TO_HEAD_WEIGHT * (3 - dist_to_opp) * 0.5

        # 5. Center preference
        center_x = board_width / 2
        center_y = board_height / 2
        dist_to_center = abs(my_head[X] - center_x) + abs(my_head[Y] - center_y)
        max_dist = center_x + center_y
        score += CENTER_WEIGHT * (max_dist - dist_to_center)

        # 6. Edge penalty
        if my_head[X] == 0 or my_head[X] == board_width - 1:
            score -= EDGE_PENALTY
        if my_head[Y] == 0 or my_head[Y] == board_height - 1:
            score -= EDGE_PENALTY

        return score

    def _flood_fill_fast(self, start_pos: typing.Dict, game_state: typing.Dict, max_depth: int = 50) -> int:
        """Fast flood fill for space evaluation."""
        board_width = game_state['board']['width']
        board_height = game_state['board']['height']

        # Build obstacle set
        obstacles = set()
        for snake in game_state['board']['snakes']:
            body = snake['body'][:-1] if len(snake['body']) > 1 else snake['body']
            for seg in body:
                obstacles.add((seg[X], seg[Y]))

        visited = set()
        queue = deque([(start_pos[X], start_pos[Y])])
        visited.add((start_pos[X], start_pos[Y]))
        count = 0

        while queue and count < max_depth:
            x, y = queue.popleft()
            count += 1

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                pos_tuple = (nx, ny)

                if pos_tuple in visited:
                    continue
                if nx < 0 or nx >= board_width or ny < 0 or ny >= board_height:
                    continue
                if pos_tuple in obstacles:
                    continue

                visited.add(pos_tuple)
                queue.append((nx, ny))

        return count

    def order_moves(self, game_state: typing.Dict, moves: typing.List[str],
                    snake_id: str, depth: int) -> typing.List[str]:
        """
        Order moves for better alpha-beta pruning.
        Best moves should be searched first.

        Args:
            game_state: Current game state
            moves: List of legal moves
            snake_id: Snake making the move
            depth: Current search depth

        Returns:
            Ordered list of moves (best first)
        """
        if not moves:
            return moves

        # If we have a best move from previous iteration, try it first
        if depth in self.best_move_at_depth and self.best_move_at_depth[depth] in moves:
            best = self.best_move_at_depth[depth]
            other_moves = [m for m in moves if m != best]
            return [best] + other_moves

        # Score each move quickly for ordering
        move_scores = []
        snake = None
        for s in game_state['board']['snakes']:
            if s['id'] == snake_id:
                snake = s
                break

        if not snake:
            return moves

        head = snake['body'][0]

        for move in moves:
            next_pos = self._get_next_position(head, move)
            score = 0

            # Prefer moves with more space
            temp_state = self.apply_move(game_state, snake_id, move)
            space = self._flood_fill_fast(next_pos, temp_state, max_depth=20)
            score += space * 10

            # Prefer moves toward food if health is low
            if snake['health'] < 50 and game_state['board']['food']:
                min_food_dist = min(
                    abs(next_pos[X] - food[X]) + abs(next_pos[Y] - food[Y])
                    for food in game_state['board']['food']
                )
                score += (20 - min_food_dist) * 5

            # Prefer center
            board_width = game_state['board']['width']
            board_height = game_state['board']['height']
            center_x = board_width / 2
            center_y = board_height / 2
            score -= abs(next_pos[X] - center_x) + abs(next_pos[Y] - center_y)

            move_scores.append((move, score))

        # Sort by score descending
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in move_scores]

    def minimax(self, game_state: typing.Dict, depth: int, alpha: float, beta: float,
                is_maximizing: bool) -> float:
        """
        Minimax with alpha-beta pruning.

        Args:
            game_state: Current game state
            depth: Remaining depth to search
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_maximizing: True if it's our turn (maximizing player)

        Returns:
            Best score achievable from this state
        """
        self.nodes_evaluated += 1

        # Check time limit
        if self.is_time_up():
            self.search_aborted = True
            return self.evaluate(game_state)

        # Check terminal state
        is_terminal, terminal_score = self.is_terminal(game_state)
        if is_terminal:
            return terminal_score

        # Check depth limit
        if depth <= 0:
            return self.evaluate(game_state)

        my_id = game_state['you']['id']

        if is_maximizing:
            # Our turn - maximize
            max_eval = -float('inf')
            legal_moves = self.get_legal_moves(game_state, my_id)

            if not legal_moves:
                return -10000  # We have no moves - we lose

            # Order moves for better pruning
            ordered_moves = self.order_moves(game_state, legal_moves, my_id, depth)

            for move in ordered_moves:
                if self.is_time_up():
                    self.search_aborted = True
                    break

                new_state = self.apply_move(game_state, my_id, move)
                # After our move, opponents move
                new_state = self.apply_opponent_moves(new_state, my_id)

                eval_score = self.minimax(new_state, depth - 1, alpha, beta, False)

                if eval_score > max_eval:
                    max_eval = eval_score

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval
        else:
            # Opponent's "turn" in our abstraction - minimize
            # Since we already applied opponent moves, we just evaluate and switch back
            # This simplified model treats opponent response as already applied
            return self.minimax(game_state, depth - 1, alpha, beta, True)

    def search(self, game_state: typing.Dict, max_depth: int = 10) -> typing.Tuple[typing.Optional[str], float, int]:
        """
        Perform iterative deepening minimax search.

        Args:
            game_state: Current game state
            max_depth: Maximum depth to search

        Returns:
            (best_move, score, depth_reached)
        """
        self.start_time = time.time()
        self.nodes_evaluated = 0
        self.search_aborted = False
        self.best_move_at_depth = {}

        my_id = game_state['you']['id']
        legal_moves = self.get_legal_moves(game_state, my_id)

        if not legal_moves:
            return None, -10000, 0

        if len(legal_moves) == 1:
            return legal_moves[0], 0, 0

        best_move = legal_moves[0]
        best_score = -float('inf')
        depth_reached = 0

        # Iterative deepening
        for depth in range(1, max_depth + 1):
            if self.is_time_up():
                break

            current_best_move = None
            current_best_score = -float('inf')

            # Order moves based on previous iteration
            ordered_moves = self.order_moves(game_state, legal_moves, my_id, depth)

            alpha = -float('inf')
            beta = float('inf')

            for move in ordered_moves:
                if self.is_time_up():
                    self.search_aborted = True
                    break

                new_state = self.apply_move(game_state, my_id, move)
                new_state = self.apply_opponent_moves(new_state, my_id)

                score = self.minimax(new_state, depth - 1, alpha, beta, True)

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move

                alpha = max(alpha, score)

            # Only update if we completed this depth
            if not self.search_aborted and current_best_move is not None:
                best_move = current_best_move
                best_score = current_best_score
                depth_reached = depth
                self.best_move_at_depth[depth] = best_move

        return best_move, best_score, depth_reached


def get_minimax_move(game_state: typing.Dict, time_limit_ms: float = 300) -> typing.Tuple[typing.Optional[str], float, int]:
    """
    Main entry point for minimax search.

    Args:
        game_state: Current game state from Battlesnake API
        time_limit_ms: Time limit for search in milliseconds

    Returns:
        (best_move, score, depth_reached)
    """
    search = MinimaxSearch(time_limit_ms=time_limit_ms)
    return search.search(game_state)
