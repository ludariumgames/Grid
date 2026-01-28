from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.base import Agent, RewardContext
from agents.random_agent import RandomAgent
from config_loader import load_rules_config
from game_engine import GameEngine
from game_setup import create_initial_game_state
from reward_engine import build_rewards_by_id
from models import Card, GameState, Joker, PlacedCard


Coord = Tuple[int, int]


def _color_letter(color: int) -> str:
    # color 1..5 -> A..E
    if 1 <= color <= 5:
        return chr(ord("A") + (color - 1))
    return "?"


def _cell_token(card: Optional[PlacedCard]) -> str:
    # 2-char tokens:
    # empty -> ..
    # normal -> <rank><colorLetter> like 3C
    # joker -> J?
    if card is None:
        return ".."
    if isinstance(card, Joker):
        return "J?"
    # Card
    return f"{card.rank}{_color_letter(card.color)}"


def _render_board(state: GameState, player_idx: int) -> List[str]:
    ps = state.players[player_idx]
    b = ps.board

    # header line
    hdr = (
        f"P{player_idx} VP={ps.vp} hand={len(ps.hand)} "
        f"skip_next={ps.skip_next_turn}"
    )
    lines = [hdr]

    # x-axis
    x_axis = "    " + " ".join(f"{x:02d}" for x in range(b.width))
    lines.append(x_axis)

    # rows
    for y in range(b.height):
        row_tokens: List[str] = []
        for x in range(b.width):
            row_tokens.append(_cell_token(b.get(x, y)))
        lines.append(f"y={y:02d} " + " ".join(row_tokens))

    return lines


def _print_state_overview(state: GameState) -> None:
    vps = [p.vp for p in state.players]
    hands = [len(p.hand) for p in state.players]
    print(
        f"STATE game={state.game_id} turn={state.turn_number} "
        f"current={state.current_player_idx} "
        f"deck={len(state.deck)} discard={len(state.discard)} "
        f"hands={hands} vp={vps}"
    )


def _card_short(c: Card) -> str:
    return f"{c.uid}(r{c.rank}{_color_letter(c.color)} s{c.shape})"


def _cards_short(cs: Sequence[Card]) -> str:
    return "[" + ", ".join(_card_short(c) for c in cs) + "]"


@dataclass
class TraceOptions:
    show_boards_before_decisions: bool = True
    show_state_overview: bool = True


class TracingAgent(Agent):
    def __init__(self, inner: Agent, opts: TraceOptions) -> None:
        self._inner = inner
        self._opts = opts

    @property
    def name(self) -> str:
        return self._inner.name

    def _pfx(self, state: GameState) -> str:
        return f"[game={state.game_id} turn={state.turn_number} p={state.current_player_idx} agent={self.name}]"

    def _maybe_print_boards(self, state: GameState) -> None:
        if not self._opts.show_boards_before_decisions:
            return
        if self._opts.show_state_overview:
            _print_state_overview(state)
        for i in range(len(state.players)):
            for line in _render_board(state, i):
                print(line)
            print("")

    def choose_placement_cell(self, state: GameState, card: Card) -> Coord:
        print(f"{self._pfx(state)} PLACE? card={_card_short(card)}")
        self._maybe_print_boards(state)
        choice = self._inner.choose_placement_cell(state, card)
        print(f"{self._pfx(state)} PLACE -> cell={choice}\n")
        return choice

    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        print(f"{self._pfx(state)} DRAFT_PICK? revealed={_cards_short(revealed_cards)}")
        self._maybe_print_boards(state)
        idx = self._inner.choose_draft_pick(state, revealed_cards)
        chosen = revealed_cards[idx] if 0 <= idx < len(revealed_cards) else None
        print(
            f"{self._pfx(state)} DRAFT_PICK -> idx={idx} chosen={_card_short(chosen) if chosen else 'INVALID'}\n"
        )
        return idx

    def choose_draft_pass(self, state: GameState, remaining_cards: Sequence[Card]) -> int:
        print(f"{self._pfx(state)} DRAFT_PASS? remaining={_cards_short(remaining_cards)}")
        self._maybe_print_boards(state)
        idx = self._inner.choose_draft_pass(state, remaining_cards)
        chosen = remaining_cards[idx] if 0 <= idx < len(remaining_cards) else None
        print(
            f"{self._pfx(state)} DRAFT_PASS -> idx={idx} pass={_card_short(chosen) if chosen else 'INVALID'}\n"
        )
        return idx

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[Any]) -> int:
        # PatternMatch лежит в models, но тип может отличаться по импорту, поэтому Any
        print(f"{self._pfx(state)} PATTERN_CHOICE? found={len(found_patterns)}")
        self._maybe_print_boards(state)
        idx = self._inner.choose_pattern_to_resolve(state, found_patterns)
        if 0 <= idx < len(found_patterns):
            m = found_patterns[idx]
            print(
                f"{self._pfx(state)} PATTERN_CHOICE -> idx={idx} pattern={m.pattern_id} vp={m.vp} reward={m.reward_id} cells={m.cells}\n"
            )
        else:
            print(f"{self._pfx(state)} PATTERN_CHOICE -> idx={idx} INVALID\n")
        return idx

    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        print(f"{self._pfx(state)} REWARD_APPLY? reward={reward_id} pattern={context.pattern_id}")
        self._maybe_print_boards(state)
        ans = self._inner.choose_apply_reward(state, reward_id, context)
        print(f"{self._pfx(state)} REWARD_APPLY -> {ans}\n")
        return ans

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        print(f"{self._pfx(state)} REWARD_PARAMS? reward={reward_id} pattern={context.pattern_id}")
        self._maybe_print_boards(state)
        params = self._inner.choose_reward_params(state, reward_id, context)
        print(f"{self._pfx(state)} REWARD_PARAMS -> {params}\n")
        return params


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="trace-one-game")
    p.add_argument("--config", default="config/game_rules_config_v0_1.json")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--agents", nargs="+", default=["random", "random"])
    p.add_argument("--no-boards", action="store_true")
    return p.parse_args()


def _resolve_config_path(config_arg: str) -> str:
    p = Path(config_arg)

    # 1) как передали
    if p.exists():
        return str(p)

    # 2) относительно корня проекта: если файл лежит в src/, то корень это ../
    here = Path(__file__).resolve()
    project_root = here.parent.parent  # Grid/
    p2 = project_root / config_arg
    if p2.exists():
        return str(p2)

    # 3) относительный вариант для случая, если тебя запустили из src/
    p3 = (here.parent / config_arg).resolve()
    if p3.exists():
        return str(p3)

    raise FileNotFoundError(
        "Config file not found.\n"
        f"Tried:\n"
        f" - {p}\n"
        f" - {p2}\n"
        f" - {p3}\n"
        "Fix: set PyCharm Working directory to project root (Grid), "
        "or pass --config ../config/game_rules_config_v0_1.json"
    )


def _build_agent(name: str, rng: random.Random, rewards_by_id: Dict[str, Any]) -> Agent:
    if name == "random":
        return RandomAgent(rng, rewards_by_id)

    raise ValueError(
        f"Unknown agent '{name}'. Supported now: random. "
        "If you added new agents, extend _build_agent in src/trace_one_game.py."
    )


def main() -> int:
    args = _parse_args()

    cfg_path = _resolve_config_path(args.config)
    cfg = load_rules_config(cfg_path)
    rewards_by_id = build_rewards_by_id(cfg)

    if len(args.agents) == 1:
        agent_names = [args.agents[0] for _ in range(args.players)]
    elif len(args.agents) == args.players:
        agent_names = list(args.agents)
    else:
        raise ValueError("Provide 1 agent name for all players, or exactly N names in --agents.")

    base_rng = random.Random(args.seed)

    # разнести RNG, чтобы было воспроизводимо и не зависело от количества логов
    deck_rng = random.Random(base_rng.getrandbits(64))
    engine_rng = random.Random(base_rng.getrandbits(64))

    state = create_initial_game_state(cfg, num_players=args.players, rng=deck_rng, game_id=1)

    agents: List[Agent] = []
    for i in range(args.players):
        aname = agent_names[i]
        arng = random.Random(base_rng.getrandbits(64))
        a = _build_agent(aname, arng, rewards_by_id)
        agents.append(
            TracingAgent(
                a,
                TraceOptions(
                    show_boards_before_decisions=not args.no_boards,
                    show_state_overview=not args.no_boards,
                ),
            )
        )

    engine = GameEngine(cfg, agents, engine_rng)
    final_state, stats, events = engine.play_game(state)

    print("\nFINAL")
    print("ended:", final_state.ended, "reason:", final_state.end_reason)
    print("turns_taken:", stats.turns_taken)
    print("deck:", len(final_state.deck), "discard:", len(final_state.discard))
    print("vp:", [p.vp for p in final_state.players])
    print("patterns_triggered_total:", sum(stats.pattern_triggers.values()))
    print("pattern_counts:", dict(stats.pattern_triggers))
    print("rewards_applied:", dict(stats.reward_applied))
    print("rewards_refused:", dict(stats.reward_refused))
    print("rewards_impossible:", dict(stats.reward_impossible))
    print("fallback_events:", len(stats.fallback_events))
    print("match_events:", len(events))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
