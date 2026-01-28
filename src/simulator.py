from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from agents.random_agent import RandomAgent
from config_loader import RulesConfig, load_rules_config
from game_engine import GameEngine
from game_setup import create_initial_game_state
from reward_engine import build_rewards_by_id


@dataclass(slots=True)
class SimulationInput:
    config_path: str
    num_players: int
    games: int
    seed: int
    agent_names: List[str]


@dataclass(slots=True)
class SimulationAggregate:
    games: int = 0
    vp_samples_by_agent: Dict[str, List[int]] = field(default_factory=dict)

    # Победа только при уникальном победителе. Ничья = 0 побед всем.
    win_count_by_agent: Dict[str, int] = field(default_factory=dict)
    unique_winner_games: int = 0
    tie_games: int = 0

    # Сколько раз срабатывал каждый паттерн (частота)
    pattern_counts: Dict[str, int] = field(default_factory=dict)

    # Суммарные VP по паттернам (по всем игрокам)
    total_vp_by_pattern: Dict[str, int] = field(default_factory=dict)

    # VP победителей по паттернам (только уникальные победы)
    winner_vp_by_pattern: Dict[str, int] = field(default_factory=dict)
    winner_triggers_by_pattern: Dict[str, int] = field(default_factory=dict)

    reward_applied: Dict[str, int] = field(default_factory=dict)
    reward_refused: Dict[str, int] = field(default_factory=dict)
    reward_impossible: Dict[str, int] = field(default_factory=dict)

    # Награды победителей (только уникальные победы)
    winner_reward_applied: Dict[str, int] = field(default_factory=dict)
    winner_reward_refused: Dict[str, int] = field(default_factory=dict)
    winner_reward_impossible: Dict[str, int] = field(default_factory=dict)

    end_reasons: Dict[str, int] = field(default_factory=dict)
    turns_total: int = 0

    fallback_total: int = 0
    fallback_by_agent: Dict[str, int] = field(default_factory=dict)
    fallback_by_decision_type: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SummaryStats:
    mean: float
    median: float
    std: float
    p90: float
    min_v: int
    max_v: int


def run_simulations(inp: SimulationInput) -> Tuple[RulesConfig, SimulationAggregate]:
    cfg = load_rules_config(inp.config_path)

    if inp.games <= 0:
        raise ValueError("games must be > 0")

    if inp.num_players < cfg.game.players.min_players or inp.num_players > cfg.game.players.max_players:
        raise ValueError(
            f"num_players must be in [{cfg.game.players.min_players}..{cfg.game.players.max_players}], got {inp.num_players}"
        )

    if len(inp.agent_names) == 1:
        agent_names = [inp.agent_names[0] for _ in range(inp.num_players)]
    elif len(inp.agent_names) == inp.num_players:
        agent_names = list(inp.agent_names)
    else:
        raise ValueError(
            f"agent_names length must be 1 or num_players={inp.num_players}, got {len(inp.agent_names)}"
        )

    rewards_by_id = build_rewards_by_id(cfg)
    agg = SimulationAggregate()

    base_rng = random.Random(inp.seed)

    for game_index in range(inp.games):
        game_id = game_index + 1

        deck_seed = base_rng.getrandbits(64)
        engine_seed = base_rng.getrandbits(64)

        deck_rng = random.Random(deck_seed)
        engine_rng = random.Random(engine_seed)

        state = create_initial_game_state(cfg, num_players=inp.num_players, rng=deck_rng, game_id=game_id)

        agents = []
        for p in range(inp.num_players):
            aname = agent_names[p]
            agent_seed = base_rng.getrandbits(64)

            if aname == "random":
                agents.append(RandomAgent(random.Random(agent_seed), rewards_by_id))
            else:
                raise ValueError(
                    f"Unsupported agent '{aname}'. Currently implemented: random. "
                    f"Next steps: center_square, center_line4, center_line5, disruptor."
                )

        engine = GameEngine(cfg, agents, engine_rng)
        final_state, stats, events = engine.play_game(state)

        agg.games += 1
        agg.turns_total += stats.turns_taken
        if final_state.end_reason is not None:
            agg.end_reasons[final_state.end_reason] = agg.end_reasons.get(final_state.end_reason, 0) + 1

        for p in range(inp.num_players):
            aname = agent_names[p]
            vp = final_state.players[p].vp
            agg.vp_samples_by_agent.setdefault(aname, []).append(vp)

        # Победа только при уникальном победителе. Ничья = 0 побед всем.
        winner = _unique_winner_index(final_state)
        if winner is not None:
            agg.unique_winner_games += 1
            aname = agent_names[winner]
            agg.win_count_by_agent[aname] = agg.win_count_by_agent.get(aname, 0) + 1
        else:
            agg.tie_games += 1

        # Частоты паттернов (триггеры)
        for pid, cnt in stats.pattern_triggers.items():
            agg.pattern_counts[pid] = agg.pattern_counts.get(pid, 0) + cnt

        # Награды (все игроки)
        for rid, cnt in stats.reward_applied.items():
            agg.reward_applied[rid] = agg.reward_applied.get(rid, 0) + cnt
        for rid, cnt in stats.reward_refused.items():
            agg.reward_refused[rid] = agg.reward_refused.get(rid, 0) + cnt
        for rid, cnt in stats.reward_impossible.items():
            agg.reward_impossible[rid] = agg.reward_impossible.get(rid, 0) + cnt

        # VP по паттернам (все игроки) и срез победителя
        for ev in events:
            agg.total_vp_by_pattern[ev.pattern_id] = agg.total_vp_by_pattern.get(ev.pattern_id, 0) + int(ev.gained_vp)

            if winner is not None and ev.player_idx == winner:
                agg.winner_vp_by_pattern[ev.pattern_id] = agg.winner_vp_by_pattern.get(ev.pattern_id, 0) + int(ev.gained_vp)
                agg.winner_triggers_by_pattern[ev.pattern_id] = agg.winner_triggers_by_pattern.get(ev.pattern_id, 0) + 1

                if ev.reward_id is not None:
                    rid = ev.reward_id
                    if ev.reward_impossible:
                        agg.winner_reward_impossible[rid] = agg.winner_reward_impossible.get(rid, 0) + 1
                    elif ev.reward_applied:
                        agg.winner_reward_applied[rid] = agg.winner_reward_applied.get(rid, 0) + 1
                    elif ev.reward_refused:
                        agg.winner_reward_refused[rid] = agg.winner_reward_refused.get(rid, 0) + 1

        # Fallback
        agg.fallback_total += len(stats.fallback_events)
        for fe in stats.fallback_events:
            agg.fallback_by_agent[fe.agent_name] = agg.fallback_by_agent.get(fe.agent_name, 0) + 1
            agg.fallback_by_decision_type[fe.decision_type] = agg.fallback_by_decision_type.get(fe.decision_type, 0) + 1

    return cfg, agg


def _unique_winner_index(state) -> Optional[int]:
    vps = [p.vp for p in state.players]
    if not vps:
        return None
    best = max(vps)
    winners = [i for i, v in enumerate(vps) if v == best]
    if len(winners) == 1:
        return winners[0]
    return None


def summarize_vp(samples: Sequence[int]) -> SummaryStats:
    if not samples:
        raise ValueError("No samples")

    xs = sorted(int(x) for x in samples)
    n = len(xs)

    mean = float(sum(xs)) / float(n)

    if n % 2 == 1:
        median = float(xs[n // 2])
    else:
        median = (float(xs[n // 2 - 1]) + float(xs[n // 2])) / 2.0

    if n >= 2:
        var = sum((float(x) - mean) ** 2 for x in xs) / float(n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0

    idx = int(math.ceil(0.9 * n) - 1)
    idx = max(0, min(n - 1, idx))
    p90 = float(xs[idx])

    return SummaryStats(
        mean=mean,
        median=median,
        std=std,
        p90=p90,
        min_v=xs[0],
        max_v=xs[-1],
    )
