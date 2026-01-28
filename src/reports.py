from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from simulator import SimulationAggregate, SimulationInput, summarize_vp


def build_report(inp: SimulationInput, agg: SimulationAggregate) -> Dict[str, Any]:
    if agg.games <= 0:
        raise ValueError("No games in aggregate")

    avg_turns = float(agg.turns_total) / float(agg.games)

    vp_stats: Dict[str, Any] = {}
    for agent_name, samples in agg.vp_samples_by_agent.items():
        s = summarize_vp(samples)
        vp_stats[agent_name] = {
            "n": len(samples),
            "mean": s.mean,
            "median": s.median,
            "std": s.std,
            "p90": s.p90,
            "min": s.min_v,
            "max": s.max_v,
        }

    # Победа только при уникальном победителе. Ничья = 0 побед всем.
    # winrate = win_count / games
    winrate: Dict[str, float] = {}
    for agent_name in agg.vp_samples_by_agent.keys():
        wins = agg.win_count_by_agent.get(agent_name, 0)
        winrate[agent_name] = float(wins) / float(agg.games)

    report: Dict[str, Any] = {
        "meta": {
            "config_path": inp.config_path,
            "players": inp.num_players,
            "games": inp.games,
            "seed": inp.seed,
            "agents": inp.agent_names,
        },
        "results": {
            "end_reasons": dict(sorted(agg.end_reasons.items(), key=lambda kv: kv[0])),
            "avg_turns": avg_turns,
            "vp_stats": vp_stats,
            "winrate": dict(sorted(winrate.items(), key=lambda kv: kv[0])),
            "pattern_counts": dict(sorted(agg.pattern_counts.items(), key=lambda kv: kv[0])),
            "rewards": {
                "applied": dict(sorted(agg.reward_applied.items(), key=lambda kv: kv[0])),
                "refused": dict(sorted(agg.reward_refused.items(), key=lambda kv: kv[0])),
                "impossible": dict(sorted(agg.reward_impossible.items(), key=lambda kv: kv[0])),
            },
            "fallback": {
                "total": agg.fallback_total,
                "by_agent": dict(sorted(agg.fallback_by_agent.items(), key=lambda kv: kv[0])),
                "by_decision_type": dict(sorted(agg.fallback_by_decision_type.items(), key=lambda kv: kv[0])),
            },
        },
    }
    return report


def print_report(report: Dict[str, Any]) -> None:
    meta = report.get("meta", {})
    res = report.get("results", {})

    print("Simulation report")
    print("config:", meta.get("config_path"))
    print("players:", meta.get("players"), "games:", meta.get("games"), "seed:", meta.get("seed"))
    print("agents:", meta.get("agents"))
    print()

    print("End reasons:", res.get("end_reasons"))
    print("Average turns:", res.get("avg_turns"))
    print()

    vp_stats = res.get("vp_stats", {})
    print("VP stats per agent:")
    for agent_name in sorted(vp_stats.keys()):
        s = vp_stats[agent_name]
        print(
            " -",
            agent_name,
            "n=",
            s.get("n"),
            "mean=",
            s.get("mean"),
            "median=",
            s.get("median"),
            "std=",
            s.get("std"),
            "p90=",
            s.get("p90"),
            "min=",
            s.get("min"),
            "max=",
            s.get("max"),
        )
    print()

    print("Winrate:", res.get("winrate"))
    print()

    print("Pattern counts:", res.get("pattern_counts"))
    print()

    rewards = res.get("rewards", {})
    print("Rewards applied:", rewards.get("applied"))
    print("Rewards refused:", rewards.get("refused"))
    print("Rewards impossible:", rewards.get("impossible"))
    print()

    fallback = res.get("fallback", {})
    print("Fallback total:", fallback.get("total"))
    print("Fallback by agent:", fallback.get("by_agent"))
    print("Fallback by decision_type:", fallback.get("by_decision_type"))


def save_report_json(path: str, report: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def load_report_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
