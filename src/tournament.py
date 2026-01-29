import sys
from pathlib import Path

# Ensure local imports work when running: python tournament.py (from src/)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import argparse
import importlib
import inspect
import random
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config_loader import load_config
from game_engine import play_game
from game_setup import create_initial_game_state


# --------------------------------- agents ---------------------------------

def _import_agent_class(agent_module_name: str):
    mod = importlib.import_module(f"agents.{agent_module_name}_agent")
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and name.endswith("Agent"):
            return obj
    raise RuntimeError(f"No *Agent class found in agents/{agent_module_name}_agent.py")


# --------------------------------- config ---------------------------------

def _resolve_config_path(config_arg: str) -> str:
    p = Path(config_arg)
    if p.exists():
        return str(p)

    # If user runs from src/, allow passing config/... relative to repo root.
    alt = (_THIS_DIR.parent / config_arg).resolve()
    if alt.exists():
        return str(alt)

    raise FileNotFoundError(
        "Config file not found.\n"
        f"Tried: {p.resolve()}\n"
        f"Also tried: {alt}\n"
        "Tip: run from src/ or pass the path relative to project root.\n"
        "Example: --config config/game_rules_config_v0_1.json"
    )


# --------------------------------- utils ----------------------------------

def _stable_u32(s: str) -> int:
    """Deterministic 32-bit hash (unlike Python's hash(), which is salted per process)."""
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


def _pair_id_unordered(a: str, b: str) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return _stable_u32(f"{lo}|{hi}")


def _fmt_hhmmss(seconds: float) -> str:
    s = max(0, int(seconds + 0.5))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


# --------------------------------- summary --------------------------------

@dataclass
class MatchupSummary:
    a0: str
    a1: str
    games: int = 0
    wins_p0: int = 0
    wins_p1: int = 0
    ties: int = 0
    sum_vp_p0: int = 0
    sum_vp_p1: int = 0
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    rewards_applied: Dict[str, int] = field(default_factory=dict)
    rewards_refused: Dict[str, int] = field(default_factory=dict)
    rewards_impossible: Dict[str, int] = field(default_factory=dict)
    fallback_events: int = 0

    def add_game(self, vp0: int, vp1: int, stats: Any) -> None:
        self.games += 1
        self.sum_vp_p0 += int(vp0)
        self.sum_vp_p1 += int(vp1)
        if vp0 > vp1:
            self.wins_p0 += 1
        elif vp1 > vp0:
            self.wins_p1 += 1
        else:
            self.ties += 1

        # Best-effort aggregation: tolerate missing stats fields.
        if stats is None:
            return

        if hasattr(stats, "pattern_triggers"):
            for k, v in dict(stats.pattern_triggers).items():
                self.pattern_counts[k] = self.pattern_counts.get(k, 0) + int(v)

        if hasattr(stats, "reward_applied"):
            for k, v in dict(stats.reward_applied).items():
                self.rewards_applied[k] = self.rewards_applied.get(k, 0) + int(v)

        if hasattr(stats, "reward_refused"):
            for k, v in dict(stats.reward_refused).items():
                self.rewards_refused[k] = self.rewards_refused.get(k, 0) + int(v)

        if hasattr(stats, "reward_impossible"):
            for k, v in dict(stats.reward_impossible).items():
                self.rewards_impossible[k] = self.rewards_impossible.get(k, 0) + int(v)

        if hasattr(stats, "fallback_events"):
            try:
                self.fallback_events += int(stats.fallback_events)
            except Exception:
                pass


@dataclass
class AgentTotals:
    games: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    sum_vp: int = 0
    sum_vp_opp: int = 0
    games_as_p0: int = 0
    games_as_p1: int = 0

    def add(self, vp: int, vp_opp: int, is_p0: bool) -> None:
        self.games += 1
        self.sum_vp += int(vp)
        self.sum_vp_opp += int(vp_opp)
        if is_p0:
            self.games_as_p0 += 1
        else:
            self.games_as_p1 += 1

        if vp > vp_opp:
            self.wins += 1
        elif vp < vp_opp:
            self.losses += 1
        else:
            self.ties += 1


# --------------------------------- game -----------------------------------

def _run_single_game(
    cfg: Any,
    rewards_by_id: Dict[str, Any],
    agent0_cls: Any,
    agent1_cls: Any,
    deck_rng: random.Random,
    game_id: int,
) -> Tuple[int, int, Any]:
    state = create_initial_game_state(cfg, deck_rng)

    agent0 = agent0_cls(player_idx=0, cfg=cfg, rewards_by_id=rewards_by_id)
    agent1 = agent1_cls(player_idx=1, cfg=cfg, rewards_by_id=rewards_by_id)

    final_state, stats = play_game(cfg, state, agents=[agent0, agent1], game_id=game_id)

    vp0 = final_state.vp[0]
    vp1 = final_state.vp[1]
    return int(vp0), int(vp1), stats


# --------------------------------- cli ------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="tournament")
    p.add_argument("--config", default="config/game_rules_config_v0_1.json")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--mirror", action="store_true", help="Play both seatings for each pair.")
    p.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N games per matchup (0 = auto).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cfg_path = _resolve_config_path(args.config)
    cfg = load_config(cfg_path)

    rewards_by_id: Dict[str, Any] = {r.reward_id: r for r in cfg.rewards}

    agent_list = list(args.agents)
    agents = {name: _import_agent_class(name) for name in agent_list}

    # Unordered pairs in a stable order.
    pairs: List[Tuple[str, str]] = []
    for i in range(len(agent_list)):
        for j in range(i + 1, len(agent_list)):
            pairs.append((agent_list[i], agent_list[j]))

    seatings: List[Tuple[str, str, str, int]] = []
    # Each element is: (pair_key, p0_agent_name, p1_agent_name, seating_idx)
    for a, b in pairs:
        pair_key = "|".join([a, b] if a <= b else [b, a])
        seatings.append((pair_key, a, b, 0))
        if args.mirror:
            seatings.append((pair_key, b, a, 1))

    total_games = len(seatings) * args.games
    progress_every = args.progress_every if args.progress_every > 0 else max(1, args.games // 100)

    print("Tournament started.")
    print(f"agents={agent_list}")
    print(f"pairs={pairs}")
    print(f"games_per_pair={args.games} seed={args.seed} mirror={args.mirror} progress_every={progress_every}")
    print("")

    results: List[MatchupSummary] = []
    totals: Dict[str, AgentTotals] = {name: AgentTotals() for name in agent_list}

    overall_done = 0
    overall_t0 = time.monotonic()

    def print_progress(
        matchup_idx: int,
        matchup_total: int,
        p0: str,
        p1: str,
        local_done: int,
        local_total: int,
    ) -> None:
        nonlocal overall_done
        now = time.monotonic()
        overall_elapsed = now - overall_t0
        overall_spg = overall_elapsed / overall_done if overall_done > 0 else 0.0
        overall_eta = overall_spg * (total_games - overall_done) if overall_spg > 0 else 0.0

        pct_local = (100.0 * local_done / local_total) if local_total > 0 else 0.0
        pct_all = (100.0 * overall_done / total_games) if total_games > 0 else 0.0

        msg = (
            f"\rmatchup {matchup_idx}/{matchup_total} | {p0} vs {p1}"
            f" | {local_done}/{local_total} ({pct_local:5.1f}%)"
            f" | overall {overall_done}/{total_games} ({pct_all:5.1f}%)"
            f" | elapsed { _fmt_hhmmss(overall_elapsed) }"
            f" | eta { _fmt_hhmmss(overall_eta) }"
        )
        print(msg, end="", flush=True)

    try:
        for matchup_idx, (pair_key, p0, p1, seating_idx) in enumerate(seatings, start=1):
            ms = MatchupSummary(a0=p0, a1=p1)
            matchup_t0 = time.monotonic()

            # Use an unordered pair id so that mirrored seatings use identical deck order for the same game index.
            pair_id = _pair_id_unordered(*pair_key.split("|", 1))

            for g in range(args.games):
                # deck_seed is independent of tournament ordering and identical for mirrored seatings.
                deck_seed = (args.seed & 0xFFFFFFFFFFFFFFFF) ^ (pair_id << 32) ^ int(g)
                deck_rng = random.Random(deck_seed)

                # game_id stays unique per game and per seating (useful for logs).
                game_id = (pair_id * 10_000_000) + (seating_idx * 1_000_000) + int(g)

                vp0, vp1, st = _run_single_game(cfg, rewards_by_id, agents[p0], agents[p1], deck_rng, game_id)
                ms.add_game(vp0, vp1, st)

                totals[p0].add(vp0, vp1, is_p0=True)
                totals[p1].add(vp1, vp0, is_p0=False)

                overall_done += 1
                if (g + 1) % progress_every == 0 or (g + 1) == args.games:
                    print_progress(matchup_idx, len(seatings), p0, p1, g + 1, args.games)

            ms_elapsed = time.monotonic() - matchup_t0
            results.append(ms)

            # Clear the progress line before printing summary.
            print("\n")

            avg0 = ms.sum_vp_p0 / ms.games if ms.games else 0.0
            avg1 = ms.sum_vp_p1 / ms.games if ms.games else 0.0
            wr0 = ms.wins_p0 / ms.games if ms.games else 0.0
            wr1 = ms.wins_p1 / ms.games if ms.games else 0.0
            tie = ms.ties / ms.games if ms.games else 0.0

            print(f"MATCHUP: {p0} (P0) vs {p1} (P1)")
            print(f"games={ms.games} winrate_p0={wr0:.3f} winrate_p1={wr1:.3f} ties={tie:.3f}")
            print(f"avg_vp_p0={avg0:.2f} avg_vp_p1={avg1:.2f} avg_diff_p0_minus_p1={(avg0 - avg1):.2f}")
            print(f"patterns_total={dict(sorted(ms.pattern_counts.items()))}")
            print(f"rewards_applied_total={dict(sorted(ms.rewards_applied.items()))}")
            print(f"rewards_refused_total={dict(sorted(ms.rewards_refused.items()))}")
            print(f"rewards_impossible_total={dict(sorted(ms.rewards_impossible.items()))}")
            print(f"fallback_events_total={ms.fallback_events}")
            print(f"elapsed_matchup={_fmt_hhmmss(ms_elapsed)}")
            print("")

    except KeyboardInterrupt:
        print("\n\nTournament interrupted by user (Ctrl+C).")
        print(f"overall_done={overall_done}/{total_games} elapsed={_fmt_hhmmss(time.monotonic() - overall_t0)}")
        print("")

        # Print whatever we have so far.
        if results:
            print("Completed matchups so far:")
            for ms in results:
                avg0 = ms.sum_vp_p0 / ms.games if ms.games else 0.0
                avg1 = ms.sum_vp_p1 / ms.games if ms.games else 0.0
                print(f" - {ms.a0} vs {ms.a1}: games={ms.games} avg_vp_p0={avg0:.2f} avg_vp_p1={avg1:.2f}")
        return 130

    # Final aggregated per-agent view (useful as a quick sanity check).
    total_elapsed = time.monotonic() - overall_t0
    print("Done.")
    print(f"elapsed_total={_fmt_hhmmss(total_elapsed)}")
    print("")

    print("AGENT TOTALS (across all matchups and seatings):")
    for name in agent_list:
        t = totals[name]
        wr = (t.wins / t.games) if t.games else 0.0
        avg_vp = (t.sum_vp / t.games) if t.games else 0.0
        avg_diff = ((t.sum_vp - t.sum_vp_opp) / t.games) if t.games else 0.0
        print(
            f" - {name}: games={t.games} winrate={wr:.3f} ties={t.ties}"
            f" avg_vp={avg_vp:.2f} avg_diff={avg_diff:.2f}"
            f" as_p0={t.games_as_p0} as_p1={t.games_as_p1}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
