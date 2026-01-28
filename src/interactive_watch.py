from __future__ import annotations

import argparse
import importlib
import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config_loader import load_rules_config
from game_engine import GameEngine
from game_setup import create_initial_game_state
from models import Board, Card, GameState, Joker, PlacedCard
from reward_engine import build_rewards_by_id

from agents.base import Agent, RewardContext


Coord = Tuple[int, int]


def _color_letter(color: int) -> str:
    # color 1..5 -> A..E
    if 1 <= color <= 5:
        return chr(ord("A") + (color - 1))
    return "?"


def _card_token(card: Card) -> str:
    # 3 символа, чтобы отличать 90 уникальных карт
    # rank(1..6) + color(A..E) + shape(1..3)
    return f"{card.rank}{_color_letter(card.color)}"


def _placed_token(pc: Optional[PlacedCard]) -> str:
    if pc is None:
        return "..."
    if isinstance(pc, Joker):
        return "J??"
    return _card_token(pc)


def _render_board_lines(board: Board) -> List[str]:
    # компактный вывод 5x5
    out: List[str] = []
    out.append("    " + " ".join(f"{x:02d}" for x in range(board.width)))
    for y in range(board.height):
        row = " ".join(_placed_token(board.get(x, y)) for x in range(board.width))
        out.append(f"y={y:02d} {row}")
    return out


def _hand_tokens(hand: Sequence[Card]) -> str:
    if not hand:
        return "[]"
    return "[" + ", ".join(_card_token(c) for c in hand) + "]"


def _resolve_config_path(config_arg: str) -> str:
    p = Path(config_arg)
    if p.exists():
        return str(p)

    # пытаемся относительно корня проекта (Grid/)
    here = Path(__file__).resolve()
    project_root = here.parent.parent  # Grid/
    p2 = project_root / config_arg
    if p2.exists():
        return str(p2)

    # если запускали из src/, то config может лежать на уровень выше
    p3 = (here.parent / config_arg).resolve()
    if p3.exists():
        return str(p3)

    raise FileNotFoundError(
        "Config file not found.\n"
        f"Tried:\n"
        f" - {p}\n"
        f" - {p2}\n"
        f" - {p3}\n"
        "Fix in PyCharm: Run -> Edit Configurations -> Working directory = project root (Grid)\n"
        "Or pass: --config ../config/game_rules_config_v0_1.json"
    )


def _import_agent_class(agent_name: str) -> type:
    name = agent_name.strip().lower()

    # маппинг алиасов
    if name in ("center_line4", "center_line4_agent", "centerline4"):
        module_name = "agents.center_line4_agent"
        preferred_class = "CenterLine4Agent"
    else:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Expected 'center_line4'."
        )

    mod = importlib.import_module(module_name)

    # 1) пробуем стандартное имя класса
    if hasattr(mod, preferred_class):
        cls = getattr(mod, preferred_class)
        if isinstance(cls, type):
            return cls

    # 2) иначе ищем любой класс-агент в модуле
    candidates: List[type] = []
    for obj in mod.__dict__.values():
        if isinstance(obj, type) and obj is not Agent and issubclass(obj, Agent):
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(
            f"Module '{module_name}' imported, but no Agent subclasses found. "
            f"Check agents/{module_name.split('.')[-1]}.py"
        )

    # если несколько, берем первый (обычно там один)
    return candidates[0]


def _instantiate_agent(
    agent_cls: type,
    *,
    rng: random.Random,
    rewards_by_id: Dict[str, Any],
    cfg: Any,
    player_idx: int,
) -> Agent:
    # пытаемся собрать kwargs по именам параметров __init__
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

        # неизвестный обязательный параметр
        if p.default is inspect._empty:
            raise TypeError(
                f"Can't instantiate {agent_cls.__name__}: unknown required __init__ param '{pname}'. "
                "Add mapping in _instantiate_agent in src/interactive_watch.py."
            )

    return agent_cls(**kwargs)  # type: ignore[misc]


def _instantiate_game_engine(cfg: Any, agents: List[Agent], rng: random.Random) -> GameEngine:
    # на случай если сигнатура конструктора менялась, подбираем параметры по именам
    sig = inspect.signature(GameEngine.__init__)
    kwargs: Dict[str, Any] = {}
    args: List[Any] = []

    # если конструктор простой (cfg, agents, rng) позиционно, это тоже пройдет, но мы аккуратнее
    # соберем kwargs по именам
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        low = pname.lower()

        if low in ("cfg", "config", "rules", "rules_config", "game_config"):
            kwargs[pname] = cfg
        elif low in ("agents", "players_agents", "agent_list"):
            kwargs[pname] = agents
        elif low in ("rng", "random", "rand", "random_state"):
            kwargs[pname] = rng
        else:
            # если параметр обязателен и мы не знаем что это, попробуем позиционно в конце
            if p.default is inspect._empty:
                args.append(None)

    try:
        return GameEngine(**kwargs)  # type: ignore[misc]
    except TypeError:
        # fallback: пробуем позиционно (cfg, agents, rng)
        return GameEngine(cfg, agents, rng)  # type: ignore[call-arg]


@dataclass
class WatchOptions:
    pause: bool = True
    show_board: str = "current"  # none | current | all


class Stepper:
    def __init__(self, opts: WatchOptions) -> None:
        self.opts = opts
        self._last_turn_key: Optional[Tuple[int, int]] = None  # (turn_number, current_player_idx)

    def pause(self, prompt: str = "Enter продолжить, q выйти: ") -> None:
        if not self.opts.pause:
            return
        try:
            s = input(prompt).strip().lower()
        except KeyboardInterrupt:
            print("\nВыход (Ctrl+C).")
            raise SystemExit(0)
        if s in ("q", "quit", "exit"):
            print("Выход.")
            raise SystemExit(0)

    def maybe_turn_header(self, state: GameState, agents_by_player: List[str]) -> None:
        key = (state.turn_number, state.current_player_idx)
        if self._last_turn_key == key:
            return
        self._last_turn_key = key

        hands = [len(p.hand) for p in state.players]
        vps = [p.vp for p in state.players]
        skips = [p.skip_next_turn for p in state.players]

        p = state.current_player_idx
        agent_name = agents_by_player[p] if 0 <= p < len(agents_by_player) else "?"

        print("")
        print(f"TURN {state.turn_number} | Player {p} ({agent_name})")
        print(f"deck={len(state.deck)} discard={len(state.discard)} hands={hands} vp={vps} skip_next={skips}")

        cp = state.current_player()
        print(f"hand P{p}: {_hand_tokens(cp.hand)}")

        self.pause()

    def show_board_if_needed(self, state: GameState, focus_player_idx: int) -> None:
        mode = self.opts.show_board.strip().lower()
        if mode == "none":
            return

        if mode == "current":
            idxs = [focus_player_idx]
        elif mode == "all":
            idxs = list(range(len(state.players)))
        else:
            return

        for i in idxs:
            ps = state.players[i]
            print(f"Board P{i} VP={ps.vp} hand={len(ps.hand)} skip_next={ps.skip_next_turn}")
            for line in _render_board_lines(ps.board):
                print(line)
            print("")


class WatchingAgent(Agent):
    def __init__(self, inner: Agent, stepper: Stepper, agents_by_player: List[str], player_idx: int) -> None:
        self._inner = inner
        self._stepper = stepper
        self._agents_by_player = agents_by_player
        self._player_idx = player_idx

        # чтобы отличать placement после draft pick от placement из hand
        self._just_picked_in_draft: bool = False

    @property
    def name(self) -> str:
        return self._inner.name

    def _turn_header(self, state: GameState) -> None:
        self._stepper.maybe_turn_header(state, self._agents_by_player)

    def choose_placement_cell(self, state: GameState, card: Card) -> Coord:
        self._turn_header(state)

        phase = "PH2 place picked card" if self._just_picked_in_draft else "PH1 place from hand/queue"
        self._just_picked_in_draft = False

        print(f"{phase}: place card {_card_token(card)}")
        self._stepper.show_board_if_needed(state, focus_player_idx=state.current_player_idx)
        self._stepper.pause("Enter чтобы увидеть выбор клетки, q выйти: ")

        cell = self._inner.choose_placement_cell(state, card)

        print(f"chosen cell: {cell} for {_card_token(card)}")
        self._stepper.pause()
        return cell

    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        self._turn_header(state)

        print("PH2 draft reveal (3 cards):")
        for i, c in enumerate(revealed_cards):
            print(f"  [{i}] {_card_token(c)} (uid={c.uid})")
        self._stepper.pause("Enter чтобы агент выбрал карту, q выйти: ")

        idx = self._inner.choose_draft_pick(state, revealed_cards)
        chosen = revealed_cards[idx] if 0 <= idx < len(revealed_cards) else None

        if chosen is None:
            print(f"pick: idx={idx} INVALID")
        else:
            print(f"pick: idx={idx} card={_card_token(chosen)} (uid={chosen.uid})")

        # следующий placement у этого игрока будет placement picked card
        self._just_picked_in_draft = True

        self._stepper.pause()
        return idx

    def choose_draft_pass(self, state: GameState, remaining_cards: Sequence[Card]) -> int:
        self._turn_header(state)

        nxt = state.player_left_of(state.current_player_idx)
        print(f"PH2 draft pass (choose 1 card to pass to player {nxt}):")
        for i, c in enumerate(remaining_cards):
            print(f"  [{i}] {_card_token(c)} (uid={c.uid})")
        self._stepper.pause("Enter чтобы агент выбрал карту для передачи, q выйти: ")

        idx = self._inner.choose_draft_pass(state, remaining_cards)
        chosen = remaining_cards[idx] if 0 <= idx < len(remaining_cards) else None

        if chosen is None:
            print(f"pass: idx={idx} INVALID")
        else:
            print(f"pass: idx={idx} card={_card_token(chosen)} (uid={chosen.uid}) -> to player {nxt}")

        self._stepper.pause()
        return idx

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[Any]) -> int:
        self._turn_header(state)

        print(f"PATTERNS found: {len(found_patterns)}")
        for i, m in enumerate(found_patterns):
            # ожидаем PatternMatch с полями pattern_id, vp, reward_id, cells
            print(f"  [{i}] {m.pattern_id} vp={m.vp} reward={m.reward_id} cells={list(m.cells)}")
        self._stepper.pause("Enter чтобы агент выбрал паттерн, q выйти: ")

        idx = self._inner.choose_pattern_to_resolve(state, found_patterns)

        if 0 <= idx < len(found_patterns):
            m = found_patterns[idx]
            print(f"chosen pattern: idx={idx} {m.pattern_id} vp={m.vp} reward={m.reward_id}")
        else:
            print(f"chosen pattern: idx={idx} INVALID")

        self._stepper.pause()
        return idx

    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        self._turn_header(state)

        print(f"REWARD offer: {reward_id} (pattern={context.pattern_id})")
        self._stepper.pause("Enter чтобы агент решил применять или нет, q выйти: ")

        ans = self._inner.choose_apply_reward(state, reward_id, context)
        print(f"apply reward? {ans}")

        self._stepper.pause()
        return ans

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        self._turn_header(state)

        print(f"REWARD params needed: {reward_id} (pattern={context.pattern_id})")
        self._stepper.pause("Enter чтобы агент выбрал параметры, q выйти: ")

        params = self._inner.choose_reward_params(state, reward_id, context)
        print(f"chosen params: {params}")

        self._stepper.pause()
        return params


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="interactive-watch")
    p.add_argument("--config", default="config/game_rules_config_v0_1.json")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--agents", nargs="+", default=["center_line4", "center_line4"])
    p.add_argument("--no-pause", action="store_true")
    p.add_argument("--show-board", choices=["none", "current", "all"], default="current")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cfg_path = _resolve_config_path(args.config)
    cfg = load_rules_config(cfg_path)
    rewards_by_id = build_rewards_by_id(cfg)

    if args.players < 2:
        raise ValueError("--players must be >= 2 for watching a match")

    if len(args.agents) == 1:
        agent_names = [args.agents[0] for _ in range(args.players)]
    elif len(args.agents) == args.players:
        agent_names = list(args.agents)
    else:
        raise ValueError("Provide 1 agent name for all players, or exactly N names in --agents.")

    base_rng = random.Random(args.seed)
    deck_rng = random.Random(base_rng.getrandbits(64))
    engine_rng = random.Random(base_rng.getrandbits(64))

    state = create_initial_game_state(cfg, num_players=args.players, rng=deck_rng, game_id=1)

    opts = WatchOptions(pause=not args.no_pause, show_board=args.show_board)
    stepper = Stepper(opts)

    # создаем агентов
    agents: List[Agent] = []
    for i in range(args.players):
        cls = _import_agent_class(agent_names[i])
        arng = random.Random(base_rng.getrandbits(64))
        inner = _instantiate_agent(cls, rng=arng, rewards_by_id=rewards_by_id, cfg=cfg, player_idx=i)
        agents.append(WatchingAgent(inner, stepper, agent_names, player_idx=i))

    engine = _instantiate_game_engine(cfg, agents, engine_rng)

    print("Interactive watch started.")
    print("Controls: Enter to continue, q + Enter to quit, Ctrl+C to quit.")
    stepper.pause()

    final_state, stats, _events = engine.play_game(state)

    print("")
    print("FINAL")
    print(f"ended={final_state.ended} reason={final_state.end_reason}")
    print(f"turns_taken={stats.turns_taken}")
    print(f"deck={len(final_state.deck)} discard={len(final_state.discard)}")
    print(f"vp={[p.vp for p in final_state.players]}")
    print(f"pattern_counts={dict(stats.pattern_triggers)}")
    print(f"rewards_applied={dict(stats.reward_applied)}")
    print(f"rewards_refused={dict(stats.reward_refused)}")
    print(f"rewards_impossible={dict(stats.reward_impossible)}")
    print(f"fallback_events={len(stats.fallback_events)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
