from __future__ import annotations

import sys
from pathlib import Path

# bootstrap sys.path so "from agents..." works when running as a script
_THIS = Path(__file__).resolve()
_SRC_DIR = _THIS.parent
_ROOT_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import argparse
import importlib
import inspect
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.base import Agent
from config_loader import load_rules_config
from game_engine import GameEngine
from game_setup import create_initial_game_state
from reward_engine import build_rewards_by_id


def _resolve_config_path(config_arg: str) -> str:
    p = Path(config_arg)
    if p.exists():
        return str(p)

    here = Path(__file__).resolve()
    project_root = here.parent.parent
    p2 = project_root / config_arg
    if p2.exists():
        return str(p2)

    p3 = (here.parent / config_arg).resolve()
    if p3.exists():
        return str(p3)

    raise FileNotFoundError(
        "Config file not found.\n"
        f"Tried:\n"
        f" - {p}\n"
        f" - {p2}\n"
        f" - {p3}\n"
        "Fix in PyCharm: Working directory = project root (Grid)\n"
        "Or pass: --config config/game_rules_config_v0_1.json"
    )


def _import_agent_class(agent_name: str) -> type:
    name = agent_name.strip().lower()

    if name in ("center_line4", "center_line4_agent", "centerline4"):
        module_name = "agents.center_line4_agent"
        preferred_class = "CenterLine4Agent"
    elif name in ("line4_h", "line4_h_agent", "h"):
        module_name = "agents.line4_h_agent"
        preferred_class = "Line4HAgent"
    elif name in ("line4_i", "line4_i_agent", "i"):
        module_name = "agents.line4_i_agent"
        preferred_class = "Line4IAgent"
    elif name in ("line4_j", "line4_j_agent", "j"):
        module_name = "agents.line4_j_agent"
        preferred_class = "Line4JAgent"
    elif name in ("random", "random_agent"):
        module_name = "agents.random_agent"
        preferred_class = "RandomAgent"
    else:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Expected one of: "
            "center_line4, line4_h, line4_i, line4_j, random."
        )

    mod = importlib.import_module(module_name)

    if hasattr(mod, preferred_class):
        cls = getattr(mod, preferred_class)
        if isinstance(cls, type):
            return cls

    candidates: List[type] = []
    for obj in mod.__dict__.values():
        if isinstance(obj, type) and obj is not Agent and issubclass(obj, Agent):
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(f"Module '{module_name}' imported, but no Agent subclasses found.")

    return candidates[0]


def _instantiate_agent(
    agent_cls: type,
    *,
    rng: random.Random,
    rewards_by_id: Dict[str, Any],
    cfg: Any,
    player_idx: int,
) -> Agent:
    sig = inspect.signature(agent_cls.__init__)
    kwargs: Dict[str, Any] = {}

    for pname, p in sig.parameters.items():
        if pname == "self":
            continue

        low = pname.lower()

        if low in ("rng", "random", "rand", "random_state"):
            kwargs[pname] = rng
            continue

        if low in ("rewards_by_id", "rewards", "reward_by_id", "reward_map", "rewards_map"):
            kwargs[pname] = rewards_by_id
            continue

        if low in ("cfg", "config", "rules", "rules_config", "game_config"):
            kwargs[pname] = cfg
            continue

        if low in ("player_idx", "player", "idx"):
            kwargs[pname] = player_idx
            continue

        if p.default is inspect._empty:
            raise TypeError(
                f"Can't instantiate {agent_cls.__name__}: unknown required __init__ param '{pname}'. "
                "Add mapping in tournament.py."
            )

    return agent_cls(**kwargs)  # type: ignore[misc]


def _instantiate_game_engine(cfg: Any, agents: List[Agent], rng: random.Random) -> GameEngine:
    sig = inspect.signature(GameEngine.__init__)
    kwargs: Dict[str, Any] = {}

    for pname, _p in sig.parameters.items():
        if pname == "self":
            continue
        low = pname.lower()
        if low in ("cfg", "config", "rules", "rules_config", "game_config"):
            kwargs[pname] = cfg
        elif low in ("agents", "players_agents", "agent_list"):
            kwargs[pname] = agents
        elif low in ("rng", "random", "rand", "random_state"):
            kwargs[pname] = rng

    try:
        return GameEngine(**kwargs)  # type: ignore[misc]
    except TypeError:
        return GameEngine(cfg, agents, rng)  # type: ignore[call-arg]


@dataclass
class MatchupStats:
    games: int = 0
    wins_p0: int = 0
    wins_p1: int = 0
    ties: int = 0
    sum_vp0: int = 0
    sum_vp1: int = 0
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    rewards_applied: Dict[str, int] = field(default_factory=dict)
    rewards_refused: Dict[str, int] = field(default_factory=dict)
    rewards_impossible: Dict[str, int] = field(default_factory=dict)
    fallback_events: int = 0

    def add_dict(self, dst: Dict[str, int], src: Any) -> None:
        if not src:
            return
        for k, v in dict(src).items():
            dst[str(k)] = dst.get(str(k), 0) + int(v)

    def add_game(self, vp0: int, vp1: int, stats: Any) -> None:
        self.games += 1
        self.sum_vp0 += int(vp0)
        self.sum_vp1 += int(vp1)

        if vp0 > vp1:
            self.wins_p0 += 1
        elif vp1 > vp0:
            self.wins_p1 += 1
        else:
            self.ties += 1

        self.add_dict(self.pattern_counts, getattr(stats, "pattern_triggers", {}))
        self.add_dict(self.rewards_applied, getattr(stats, "reward_applied", {}))
        self.add_dict(self.rewards_refused, getattr(stats, "reward_refused", {}))
        self.add_dict(self.rewards_impossible, getattr(stats, "reward_impossible", {}))
        fb = getattr(stats, "fallback_events", None)
        if fb is not None:
            try:
                self.fallback_events += len(fb)
            except TypeError:
                self.fallback_events += int(fb)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="tournament")
    p.add_argument("--config", default="config/game_rules_config_v0_1.json")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--mirror", action="store_true", help="Also run mirrored seating (A vs B and B vs A)")
    return p.parse_args()


def _run_single_game(cfg: Any, rewards_by_id: Dict[str, Any], a0: str, a1: str, base_seed: int, game_id: int) -> Tuple[int, int, Any]:
    base_rng = random.Random((base_seed << 20) ^ game_id)

    deck_rng = random.Random(base_rng.getrandbits(64))
    engine_rng = random.Random(base_rng.getrandbits(64))

    state = create_initial_game_state(cfg, num_players=2, rng=deck_rng, game_id=game_id)

    agents: List[Agent] = []
    for idx, aname in enumerate([a0, a1]):
        cls = _import_agent_class(aname)
        arng = random.Random(base_rng.getrandbits(64))
        inner = _instantiate_agent(cls, rng=arng, rewards_by_id=rewards_by_id, cfg=cfg, player_idx=idx)
        agents.append(inner)

    engine = _instantiate_game_engine(cfg, agents, engine_rng)
    final_state, stats, _events = engine.play_game(state)

    vp0 = int(final_state.players[0].vp)
    vp1 = int(final_state.players[1].vp)
    return vp0, vp1, stats


def main() -> int:
    args = _parse_args()
    cfg_path = _resolve_config_path(args.config)
    cfg = load_rules_config(cfg_path)
    rewards_by_id = build_rewards_by_id(cfg)

    agent_list = [a.strip() for a in args.agents if a.strip()]
    if len(agent_list) < 2:
        raise ValueError("--agents must include at least 2 agent names")

    pairs: List[Tuple[str, str]] = []
    for i in range(len(agent_list)):
        for j in range(i + 1, len(agent_list)):
            pairs.append((agent_list[i], agent_list[j]))
    if args.mirror:
        mirrored: List[Tuple[str, str]] = []
        for (a, b) in pairs:
            mirrored.append((a, b))
            mirrored.append((b, a))
        pairs = mirrored

    print("Tournament started.")
    print(f"agents={agent_list}")
    print(f"pairs={pairs}")
    print(f"games_per_pair={args.games} seed={args.seed}")

    for (a0, a1) in pairs:
        ms = MatchupStats()
        for g in range(args.games):
            vp0, vp1, st = _run_single_game(cfg, rewards_by_id, a0, a1, args.seed, game_id=(hash((a0, a1)) & 0xFFFF) * 100000 + g)
            ms.add_game(vp0, vp1, st)

        avg0 = ms.sum_vp0 / ms.games if ms.games else 0.0
        avg1 = ms.sum_vp1 / ms.games if ms.games else 0.0
        wr0 = ms.wins_p0 / ms.games if ms.games else 0.0
        wr1 = ms.wins_p1 / ms.games if ms.games else 0.0
        tie = ms.ties / ms.games if ms.games else 0.0

        print("")
        print(f"MATCHUP: {a0} (P0) vs {a1} (P1)")
        print(f"games={ms.games} winrate_p0={wr0:.3f} winrate_p1={wr1:.3f} ties={tie:.3f}")
        print(f"avg_vp_p0={avg0:.2f} avg_vp_p1={avg1:.2f} avg_diff_p0_minus_p1={(avg0 - avg1):.2f}")
        print(f"patterns_total={dict(sorted(ms.pattern_counts.items()))}")
        print(f"rewards_applied_total={dict(sorted(ms.rewards_applied.items()))}")
        print(f"rewards_refused_total={dict(sorted(ms.rewards_refused.items()))}")
        print(f"rewards_impossible_total={dict(sorted(ms.rewards_impossible.items()))}")
        print(f"fallback_events_total={ms.fallback_events}")

    print("")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
