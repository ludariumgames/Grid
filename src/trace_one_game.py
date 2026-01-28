from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from agents.base import Agent, RewardContext
from agents.random_agent import RandomAgent
from config_loader import load_rules_config
from game_engine import GameEngine
from game_setup import create_initial_game_state
from reward_engine import build_rewards_by_id
from models import Card, Coord, GameState, PatternMatch


def _card_short(c: Card) -> str:
    return f"{c.uid}(r{c.rank} c{c.color} s{c.shape})"


def _cards_short(cs: Sequence[Card]) -> str:
    return "[" + ", ".join(_card_short(c) for c in cs) + "]"


@dataclass
class TraceOptions:
    show_turn_header: bool = True


class TracingAgent(Agent):
    def __init__(self, inner: Agent, opts: TraceOptions) -> None:
        self._inner = inner
        self._opts = opts

    @property
    def name(self) -> str:
        return self._inner.name

    def _pfx(self, state: GameState) -> str:
        return f"[game={state.game_id} turn={state.turn_number} p={state.current_player_idx} agent={self.name}]"

    def choose_placement_cell(self, state: GameState, card: Card) -> Coord:
        choice = self._inner.choose_placement_cell(state, card)
        print(f"{self._pfx(state)} PLACE card={_card_short(card)} -> cell={choice}")
        return choice

    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        idx = self._inner.choose_draft_pick(state, revealed_cards)
        chosen = revealed_cards[idx] if 0 <= idx < len(revealed_cards) else None
        print(f"{self._pfx(state)} DRAFT_PICK revealed={_cards_short(revealed_cards)} -> idx={idx} chosen={_card_short(chosen) if chosen else 'INVALID'}")
        return idx

    def choose_draft_pass(self, state: GameState, remaining_cards: Sequence[Card]) -> int:
        idx = self._inner.choose_draft_pass(state, remaining_cards)
        chosen = remaining_cards[idx] if 0 <= idx < len(remaining_cards) else None
        print(f"{self._pfx(state)} DRAFT_PASS remaining={_cards_short(remaining_cards)} -> idx={idx} pass={_card_short(chosen) if chosen else 'INVALID'}")
        return idx

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[PatternMatch]) -> int:
        idx = self._inner.choose_pattern_to_resolve(state, found_patterns)
        if 0 <= idx < len(found_patterns):
            m = found_patterns[idx]
            print(f"{self._pfx(state)} PATTERN_CHOICE found={len(found_patterns)} -> idx={idx} pattern={m.pattern_id} vp={m.vp} reward={m.reward_id} cells={m.cells}")
        else:
            print(f"{self._pfx(state)} PATTERN_CHOICE found={len(found_patterns)} -> idx={idx} INVALID")
        return idx

    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        ans = self._inner.choose_apply_reward(state, reward_id, context)
        print(f"{self._pfx(state)} REWARD_APPLY? reward={reward_id} pattern={context.pattern_id} -> {ans}")
        return ans

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        params = self._inner.choose_reward_params(state, reward_id, context)
        print(f"{self._pfx(state)} REWARD_PARAMS reward={reward_id} -> {params}")
        return params


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="trace-one-game")
    p.add_argument("--config", required=True)
    p.add_argument("--players", type=int, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--agents", nargs="+", required=True)
    return p.parse_args()


def _build_agent(name: str, rng: random.Random, rewards_by_id: Dict[str, Any]) -> Agent:
    if name == "random":
        return RandomAgent(rng, rewards_by_id)
    raise ValueError(f"Unsupported agent '{name}' for tracing. Add it to _build_agent in src/trace_one_game.py.")


def main() -> int:
    args = _parse_args()

    cfg = load_rules_config(args.config)
    rewards_by_id = build_rewards_by_id(cfg)

    if len(args.agents) == 1:
        agent_names = [args.agents[0] for _ in range(args.players)]
    elif len(args.agents) == args.players:
        agent_names = list(args.agents)
    else:
        raise ValueError("Provide 1 agent name for all players, or exactly N names.")

    base_rng = random.Random(args.seed)
    deck_rng = random.Random(base_rng.getrandbits(64))
    engine_rng = random.Random(base_rng.getrandbits(64))

    state = create_initial_game_state(cfg, num_players=args.players, rng=deck_rng, game_id=1)

    agents: List[Agent] = []
    for i in range(args.players):
        aname = agent_names[i]
        arng = random.Random(base_rng.getrandbits(64))
        a = _build_agent(aname, arng, rewards_by_id)
        agents.append(TracingAgent(a, TraceOptions()))

    engine = GameEngine(cfg, agents, engine_rng)
    final_state, stats, events = engine.play_game(state)

    print("\nFINAL")
    print("ended:", final_state.ended, "reason:", final_state.end_reason)
    print("turns_taken:", stats.turns_taken)
    print("deck:", len(final_state.deck), "discard:", len(final_state.discard))
    print("vp:", [p.vp for p in final_state.players])
    print("patterns_triggered_total:", sum(stats.pattern_triggers.values()))
    print("pattern_counts:", dict(stats.pattern_triggers))
    print("reward_applied:", dict(stats.reward_applied))
    print("reward_refused:", dict(stats.reward_refused))
    print("reward_impossible:", dict(stats.reward_impossible))
    print("fallback_events:", len(stats.fallback_events))
    if stats.fallback_events:
        print("fallback_sample:", stats.fallback_events[0])

    # events может отличаться по структуре в твоей ветке, но оставим как есть
    print("match_events:", len(events))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
