#!/usr/bin/env python3
"""
Battlesnake Version Benchmarking Tool

This script benchmarks your current snake (your_snake) against the old version (your_old_snake).
It runs multiple games and generates a comprehensive report.

Usage:
    python benchmark_snakes.py [--iterations N] [--workers N]

The script will:
1. Start both snake servers automatically
2. Run N games (default 500)
3. Generate detailed statistics and a report
4. Save results to benchmarks/ directory
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import sympy
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.go_utils import check_and_build_rules_cli


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    iterations: int = 500
    workers: int = 8
    width: int = 11
    height: int = 11
    new_snake_port: int = 7200
    old_snake_port: int = 7201
    new_snake_name: str = "NewSnake"
    old_snake_name: str = "OldSnake"
    new_snake_dir: str = "your_snake"
    old_snake_dir: str = "your_old_snake"
    timeout_ms: int = 10000


@dataclass
class GameResult:
    """Result of a single game"""
    game_num: int
    winner: Optional[str]  # "new", "old", "draw", or None for error
    turns: int = 0
    new_snake_length: int = 0
    old_snake_length: int = 0
    new_snake_health: int = 0
    old_snake_health: int = 0
    death_reason_new: str = ""
    death_reason_old: str = ""
    error: str = ""


@dataclass
class BenchmarkStats:
    """Aggregated benchmark statistics"""
    total_games: int = 0
    new_wins: int = 0
    old_wins: int = 0
    draws: int = 0
    errors: int = 0

    total_turns: int = 0
    new_total_length: int = 0
    old_total_length: int = 0

    new_death_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    old_death_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    game_results: List[GameResult] = field(default_factory=list)

    @property
    def new_win_rate(self) -> float:
        valid_games = self.total_games - self.errors
        return self.new_wins / valid_games * 100 if valid_games > 0 else 0

    @property
    def old_win_rate(self) -> float:
        valid_games = self.total_games - self.errors
        return self.old_wins / valid_games * 100 if valid_games > 0 else 0

    @property
    def draw_rate(self) -> float:
        valid_games = self.total_games - self.errors
        return self.draws / valid_games * 100 if valid_games > 0 else 0

    @property
    def avg_turns(self) -> float:
        valid_games = self.total_games - self.errors
        return self.total_turns / valid_games if valid_games > 0 else 0

    @property
    def avg_new_length(self) -> float:
        valid_games = self.total_games - self.errors
        return self.new_total_length / valid_games if valid_games > 0 else 0

    @property
    def avg_old_length(self) -> float:
        valid_games = self.total_games - self.errors
        return self.old_total_length / valid_games if valid_games > 0 else 0


class SnakeServerManager:
    """Manages starting and stopping snake servers"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.processes: List[subprocess.Popen] = []

    def start_servers(self) -> bool:
        """Start both snake servers"""
        print("\n🐍 Starting snake servers...")

        # Start new snake server
        new_snake_process = self._start_server(
            self.config.new_snake_dir,
            self.config.new_snake_port,
            "NewSnake"
        )
        if not new_snake_process:
            return False
        self.processes.append(new_snake_process)

        # Start old snake server
        old_snake_process = self._start_server(
            self.config.old_snake_dir,
            self.config.old_snake_port,
            "OldSnake"
        )
        if not old_snake_process:
            self.stop_servers()
            return False
        self.processes.append(old_snake_process)

        # Wait for servers to be ready
        print("   Waiting for servers to start...")
        time.sleep(2)

        # Verify servers are responding
        if not self._check_server(self.config.new_snake_port, "NewSnake"):
            self.stop_servers()
            return False

        if not self._check_server(self.config.old_snake_port, "OldSnake"):
            self.stop_servers()
            return False

        print("   ✅ Both servers are running!\n")
        return True

    def _start_server(self, snake_dir: str, port: int, name: str) -> Optional[subprocess.Popen]:
        """Start a single snake server"""
        snake_path = PROJECT_ROOT / snake_dir

        if not snake_path.exists():
            print(f"   ❌ Error: {snake_dir} directory not found!")
            return None

        env = os.environ.copy()
        env["PORT"] = str(port)
        env["PYTHONPATH"] = str(snake_path)

        try:
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=snake_path,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"   Started {name} on port {port} (PID: {process.pid})")
            return process
        except Exception as e:
            print(f"   ❌ Error starting {name}: {e}")
            return None

    def _check_server(self, port: int, name: str) -> bool:
        """Check if a server is responding"""
        import urllib.request
        import urllib.error

        url = f"http://localhost:{port}/"
        max_retries = 10

        for i in range(max_retries):
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError):
                time.sleep(0.5)

        print(f"   ❌ {name} server not responding on port {port}")
        return False

    def stop_servers(self):
        """Stop all snake servers"""
        print("\n🛑 Stopping snake servers...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass
        self.processes.clear()
        print("   ✅ Servers stopped\n")


def _determine_death_cause(snake_data: dict, game_state: dict, config: BenchmarkConfig) -> str:
    """Try to determine why a snake died based on its final position"""
    if not snake_data:
        return "unknown"

    head = snake_data.get("head", {})
    health = snake_data.get("health", 100)
    body = snake_data.get("body", [])

    # Check if starved
    if health <= 0:
        return "starvation"

    # Check if out of bounds
    x, y = head.get("x", 0), head.get("y", 0)
    if x < 0 or x >= config.width or y < 0 or y >= config.height:
        return "out-of-bounds"

    # Check if collided with self
    if len(body) > 1:
        for segment in body[1:]:
            if segment.get("x") == x and segment.get("y") == y:
                return "self-collision"

    # Check if collided with other snake
    for snake in game_state.get("board", {}).get("snakes", []):
        if snake.get("name") == snake_data.get("name"):
            continue
        for segment in snake.get("body", []):
            if segment.get("x") == x and segment.get("y") == y:
                return "snake-collision"

    # Head-to-head collision (most likely if we got here)
    return "head-collision"


def run_single_game(game_num: int, seed: str, config: BenchmarkConfig, output_dir: Path) -> GameResult:
    """Run a single game and return the result"""
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    output_file = games_dir / f"game_{game_num}.json"

    cmd = [
        str(PROJECT_ROOT / "rules" / "battlesnake"),
        "play",
        "-W", str(config.width),
        "-H", str(config.height),
        "-n", config.new_snake_name,
        "-u", f"http://localhost:{config.new_snake_port}",
        "-n", config.old_snake_name,
        "-u", f"http://localhost:{config.old_snake_port}",
        "-r", seed,
        "-o", str(output_file),
        "--timeout", str(config.timeout_ms),
    ]

    result = GameResult(game_num=game_num, winner=None)

    try:
        proc_result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if proc_result.returncode != 0:
            # Check which snake failed
            stderr = proc_result.stderr.lower()
            if str(config.new_snake_port) in stderr:
                result.winner = "old"
                result.error = "new_snake_timeout"
            elif str(config.old_snake_port) in stderr:
                result.winner = "new"
                result.error = "old_snake_timeout"
            else:
                result.error = f"unknown_error: {proc_result.stderr[:200]}"
            return result

        # Parse game result
        if output_file.exists():
            with open(output_file) as f:
                content = f.read().strip()
                lines = content.split('\n')
                if lines:
                    # The last line contains winner info (winnerId, winnerName, isDraw)
                    # The second-to-last line contains the final game state with turn/snake data
                    final_line = json.loads(lines[-1])

                    # Get turn count and snake stats from second-to-last line (if available)
                    if len(lines) >= 2:
                        game_state = json.loads(lines[-2])
                        result.turns = game_state.get("turn", 0)

                        # Get snake states from the final game state
                        for snake in game_state.get("board", {}).get("snakes", []):
                            if snake.get("name") == config.new_snake_name:
                                result.new_snake_length = snake.get("length", len(snake.get("body", [])))
                                result.new_snake_health = snake.get("health", 0)
                            elif snake.get("name") == config.old_snake_name:
                                result.old_snake_length = snake.get("length", len(snake.get("body", [])))
                                result.old_snake_health = snake.get("health", 0)

                        # Determine death reasons by checking which snakes are missing/eliminated
                        # A snake that lost is not in the final snakes list
                        final_snake_names = [s.get("name") for s in game_state.get("board", {}).get("snakes", [])]

                        if config.new_snake_name not in final_snake_names:
                            # New snake was eliminated - determine cause
                            you_data = game_state.get("you", {})
                            if you_data.get("name") == config.new_snake_name:
                                # Check elimination cause from position
                                result.death_reason_new = _determine_death_cause(you_data, game_state, config)
                            else:
                                result.death_reason_new = "eliminated"

                        if config.old_snake_name not in final_snake_names:
                            result.death_reason_old = "eliminated"

                    # Determine winner from the final line
                    if final_line.get("isDraw"):
                        result.winner = "draw"
                    elif final_line.get("winnerName") == config.new_snake_name:
                        result.winner = "new"
                    elif final_line.get("winnerName") == config.old_snake_name:
                        result.winner = "old"

    except subprocess.TimeoutExpired:
        result.error = "game_timeout"
    except Exception as e:
        result.error = str(e)

    return result


class BenchmarkRunner:
    """Main benchmark runner"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.stats = BenchmarkStats()
        self.stats_lock = Lock()

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = PROJECT_ROOT / "benchmarks" / f"benchmark_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> BenchmarkStats:
        """Run the full benchmark"""
        # Check battlesnake CLI
        if not check_and_build_rules_cli():
            print("❌ Battlesnake CLI not available. Please build it first.")
            print("   Run: cd rules && make build")
            sys.exit(1)

        # Start servers
        server_manager = SnakeServerManager(self.config)
        if not server_manager.start_servers():
            print("❌ Failed to start snake servers")
            sys.exit(1)

        try:
            self._run_games()
        finally:
            server_manager.stop_servers()

        return self.stats

    def _run_games(self):
        """Run all benchmark games"""
        print("=" * 70)
        print("     BATTLESNAKE VERSION BENCHMARK")
        print(f"     {self.config.new_snake_name} (new) vs {self.config.old_snake_name} (old)")
        print(f"     Running {self.config.iterations} games with {self.config.workers} workers")
        print(f"     Board: {self.config.width}x{self.config.height}")
        print("=" * 70)
        print()

        # Generate game parameters with prime seeds
        game_params = []
        last_prime = 100
        for i in range(self.config.iterations):
            next_prime = sympy.nextprime(last_prime)
            game_params.append((i, str(next_prime)))
            last_prime = next_prime

        # Run games in parallel
        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {
                executor.submit(
                    run_single_game,
                    game_num, seed, self.config, self.output_dir
                ): game_num
                for game_num, seed in game_params
            }

            bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            with tqdm(total=self.config.iterations, desc="Running games", bar_format=bar_fmt) as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    with self.stats_lock:
                        self.stats.total_games += 1
                        self.stats.game_results.append(result)

                        if result.error and result.winner is None:
                            self.stats.errors += 1
                        elif result.winner == "new":
                            self.stats.new_wins += 1
                        elif result.winner == "old":
                            self.stats.old_wins += 1
                        elif result.winner == "draw":
                            self.stats.draws += 1

                        self.stats.total_turns += result.turns
                        self.stats.new_total_length += result.new_snake_length
                        self.stats.old_total_length += result.old_snake_length

                        if result.death_reason_new:
                            self.stats.new_death_reasons[result.death_reason_new] += 1
                        if result.death_reason_old:
                            self.stats.old_death_reasons[result.death_reason_old] += 1

                    pbar.set_postfix({
                        "New": self.stats.new_wins,
                        "Old": self.stats.old_wins,
                        "Draw": self.stats.draws,
                    })
                    pbar.update(1)

    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report"""
        s = self.stats

        report = []
        report.append("\n" + "=" * 70)
        report.append("                    BENCHMARK REPORT")
        report.append("=" * 70)

        # Overall results
        report.append("\n📊 OVERALL RESULTS")
        report.append("-" * 40)
        report.append(f"   Total Games:     {s.total_games}")
        report.append(f"   Valid Games:     {s.total_games - s.errors}")
        report.append(f"   Errors:          {s.errors}")
        report.append("")

        # Win rates
        report.append("🏆 WIN RATES")
        report.append("-" * 40)
        report.append(f"   {self.config.new_snake_name} (new):  {s.new_wins:4d} wins ({s.new_win_rate:5.1f}%)")
        report.append(f"   {self.config.old_snake_name} (old):  {s.old_wins:4d} wins ({s.old_win_rate:5.1f}%)")
        report.append(f"   Draws:           {s.draws:4d}      ({s.draw_rate:5.1f}%)")
        report.append("")

        # Winner determination
        if s.new_wins > s.old_wins:
            improvement = ((s.new_wins / max(s.old_wins, 1)) - 1) * 100
            report.append(f"   🎉 NEW SNAKE WINS! (+{improvement:.1f}% improvement)")
        elif s.old_wins > s.new_wins:
            regression = ((s.old_wins / max(s.new_wins, 1)) - 1) * 100
            report.append(f"   ⚠️  OLD SNAKE WINS! ({regression:.1f}% regression)")
        else:
            report.append("   🤝 IT'S A TIE!")
        report.append("")

        # Performance stats
        report.append("📈 PERFORMANCE STATISTICS")
        report.append("-" * 40)
        report.append(f"   Average Game Length:    {s.avg_turns:.1f} turns")
        report.append(f"   Avg New Snake Length:   {s.avg_new_length:.1f}")
        report.append(f"   Avg Old Snake Length:   {s.avg_old_length:.1f}")
        report.append("")

        # Death reasons for new snake
        if s.new_death_reasons:
            report.append(f"💀 {self.config.new_snake_name} DEATH REASONS")
            report.append("-" * 40)
            for reason, count in sorted(s.new_death_reasons.items(), key=lambda x: -x[1]):
                pct = count / (s.total_games - s.errors) * 100 if (s.total_games - s.errors) > 0 else 0
                report.append(f"   {reason:30s} {count:4d} ({pct:5.1f}%)")
            report.append("")

        # Death reasons for old snake
        if s.old_death_reasons:
            report.append(f"💀 {self.config.old_snake_name} DEATH REASONS")
            report.append("-" * 40)
            for reason, count in sorted(s.old_death_reasons.items(), key=lambda x: -x[1]):
                pct = count / (s.total_games - s.errors) * 100 if (s.total_games - s.errors) > 0 else 0
                report.append(f"   {reason:30s} {count:4d} ({pct:5.1f}%)")
            report.append("")

        # Statistical significance
        valid_games = s.total_games - s.errors
        if valid_games >= 30:
            # Simple binomial confidence interval approximation
            p_new = s.new_wins / valid_games if valid_games > 0 else 0
            se = (p_new * (1 - p_new) / valid_games) ** 0.5 if valid_games > 0 else 0
            ci_low = max(0, p_new - 1.96 * se)
            ci_high = min(1, p_new + 1.96 * se)

            report.append("📉 STATISTICAL ANALYSIS")
            report.append("-" * 40)
            report.append(f"   New Snake Win Rate: {p_new*100:.1f}%")
            report.append(f"   95% Confidence Interval: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

            if ci_low > 0.5:
                report.append("   ✅ New snake is SIGNIFICANTLY BETTER (p < 0.05)")
            elif ci_high < 0.5:
                report.append("   ❌ New snake is SIGNIFICANTLY WORSE (p < 0.05)")
            else:
                report.append("   ⚖️  No statistically significant difference")
            report.append("")

        report.append("=" * 70)
        report.append(f"   Results saved to: {self.output_dir}")
        report.append("=" * 70)

        return "\n".join(report)

    def save_results(self):
        """Save benchmark results to files"""
        # Save summary JSON
        summary = {
            "config": {
                "iterations": self.config.iterations,
                "workers": self.config.workers,
                "width": self.config.width,
                "height": self.config.height,
                "new_snake_name": self.config.new_snake_name,
                "old_snake_name": self.config.old_snake_name,
            },
            "results": {
                "total_games": self.stats.total_games,
                "new_wins": self.stats.new_wins,
                "old_wins": self.stats.old_wins,
                "draws": self.stats.draws,
                "errors": self.stats.errors,
                "new_win_rate": self.stats.new_win_rate,
                "old_win_rate": self.stats.old_win_rate,
                "avg_turns": self.stats.avg_turns,
                "avg_new_length": self.stats.avg_new_length,
                "avg_old_length": self.stats.avg_old_length,
            },
            "death_reasons": {
                "new_snake": dict(self.stats.new_death_reasons),
                "old_snake": dict(self.stats.old_death_reasons),
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed report
        report = self.generate_report()
        with open(self.output_dir / "report.txt", "w") as f:
            f.write(report)

        print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark your new snake against the old version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_snakes.py                    # Run 500 games with 8 workers
    python benchmark_snakes.py --iterations 100  # Quick test with 100 games
    python benchmark_snakes.py --workers 6       # Use 6 parallel workers
        """
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=500,
        help="Number of games to run (default: 500)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=11,
        help="Board width (default: 11)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=11,
        help="Board height (default: 11)"
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        iterations=args.iterations,
        workers=args.workers,
        width=args.width,
        height=args.height,
    )

    runner = BenchmarkRunner(config)
    runner.run()
    runner.save_results()


if __name__ == "__main__":
    main()
