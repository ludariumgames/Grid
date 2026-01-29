from __future__ import annotations

import sys
from pathlib import Path

# --- bootstrap: allow running as a script from anywhere (fixes ModuleNotFoundError: agents) ---
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

from agents.base import Agent, RewardContext
from config_loader import load_rules_config
from game_engine import GameEngine
from game_setup import create_initial_game_state
from models import Board, Card, GameState, Joker, PlacedCard
from reward_engine import build_rewards_by_id

Coord = Tuple[int, int]


def _color_letter(color: int) -> str:
    # color 1..5 -> A..E
    if 1 <= color <= 5:
        return chr(ord("A") + (color - 1))
    return "?"


def _card_token(card: Card) -> str:
    return f"{card.rank}{_color_letter(card.color)}"


def _placed_token(pc: Optional[PlacedCard], mark: Optional[str]) -> str:
    # fixed width 3 chars
    # empty: " .."
    # card:  " 6A"
    # marked: "+6A" or "*6A"
    if pc is None:
        return " .."
    if isinstance(pc, Joker):
        base = "J?"
    else:
        base = _card_token(pc)
    if mark:
        return f"{mark}{base}"
    return f" {base}"


def _render_board(board: Board, highlights: Optional[Sequence[Coord]] = None, mark: str = "+") -> List[str]:
    hs = set(highlights or [])
    out: List[str] = []
    for y in range(board.height):
        row_tokens: List[str] = []
        for x in range(board.width):
            row_tokens.append(_placed_token(board.get(x, y), mark if (x, y) in hs else None))
        out.append("  ".join(row_tokens))
    return out


def _hand_tokens(hand: Sequence[Card]) -> str:
    if not hand:
        return "[]"
    return "[" + ", ".join(_card_token(c) for c in hand) + "]"


def _resolve_config_path(config_arg: str) -> str:
    p = Path(config_arg)
    if p.exists():
        return str(p)

    here = Path(__file__).resolve()
    project_root = here.parent.parent  # Grid/
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
        "Fix in PyCharm: Run -> Edit Configurations -> Working directory = project root (Grid)\n"
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
                "Add mapping in _instantiate_agent in src/interactive_watch.py."
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
class TurnDraft:
    revealed: List[str] = field(default_factory=list)
    pick: Optional[str] = None
    pass_to_idx: Optional[int] = None
    passed: Optional[str] = None
    discarded: Optional[str] = None


@dataclass
class TurnPlacement:
    card: str
    cell: Coord


@dataclass
class PendingPattern:
    player_idx: int
    turn_number: int
    pattern_id: str
    vp: int
    reward_id: Optional[str]
    cells: List[Coord]
    apply_reward: Optional[bool] = None
    reward_params: Optional[Dict[str, Any]] = None
    board_before_lines: List[str] = field(default_factory=list)
    before_snapshot: Dict[Coord, Optional[str]] = field(default_factory=dict)


@dataclass
class TurnContext:
    turn_number: int
    player_idx: int
    agent_name: str
    start_deck: int
    start_discard: int
    start_hand: List[str]
    start_incoming: Optional[str]
    start_vps: List[int]
    draft: TurnDraft = field(default_factory=TurnDraft)
    placements: List[TurnPlacement] = field(default_factory=list)


class WatchUI:
    def __init__(self, pause_each_turn: bool) -> None:
        self.pause_each_turn = pause_each_turn

    def pause(self) -> None:
        if not self.pause_each_turn:
            return
        try:
            s = input("Enter продолжить, q выйти: ").strip().lower()
        except KeyboardInterrupt:
            print("\nВыход (Ctrl+C).")
            raise SystemExit(0)
        if s in ("q", "quit", "exit"):
            print("Выход.")
            raise SystemExit(0)

    def print_block(self, lines: Sequence[str], pause_after: bool = True) -> None:
        for ln in lines:
            print(ln)
        if pause_after:
            self.pause()


def _snapshot_board_tokens(board: Board) -> Dict[Coord, Optional[str]]:
    snap: Dict[Coord, Optional[str]] = {}
    for y in range(board.height):
        for x in range(board.width):
            v = board.get(x, y)
            if v is None:
                snap[(x, y)] = None
            elif isinstance(v, Joker):
                snap[(x, y)] = "J?"
            else:
                snap[(x, y)] = _card_token(v)
    return snap


class TurnLogger:
    def __init__(self, ui: WatchUI, agents_by_player: List[str]) -> None:
        self.ui = ui
        self.agents_by_player = agents_by_player

        self.current_key: Optional[Tuple[int, int]] = None  # (turn_number, player_idx)
        self.current_ctx: Optional[TurnContext] = None

        # карта, которую логически "передали" игроку, чтобы показать в начале его хода
        self.incoming_by_player: Dict[int, str] = {}

        # паттерн, который только что выбрали и сейчас будет награда/очистка
        self.pending_pattern: Optional[PendingPattern] = None

    def _flush_pending_pattern_if_ready(self, state: GameState) -> None:
        if self.pending_pattern is None:
            return

        pp = self.pending_pattern
        pidx = pp.player_idx
        ps = state.players[pidx]

        # ВАЖНО: движок может очистить паттерн не сразу.
        # Печатаем блок только когда паттерн-клетки реально пустые.
        for (x, y) in pp.cells:
            if ps.board.get(x, y) is not None:
                return

        lines: List[str] = []
        lines.append("")
        lines.append(f"PATTERN RESOLVED | Player {pidx} turn {pp.turn_number}")
        lines.append(f"pattern={pp.pattern_id} vp={pp.vp} reward={pp.reward_id} cells={pp.cells}")

        if pp.board_before_lines:
            lines.append("board before clear (pattern cells marked with '*'):")
            lines.extend(pp.board_before_lines)

        if pp.reward_id is not None:
            lines.append(f"reward decision: apply={pp.apply_reward}")
            if pp.reward_params is not None:
                lines.append(f"reward params: {pp.reward_params}")

            if pp.apply_reward is False:
                lines.append("reward result: skipped_by_agent")
            elif pp.apply_reward is True:
                after_snap = _snapshot_board_tokens(ps.board)
                pattern_cells = set(pp.cells)
                changed_outside = False
                for coord, before_val in pp.before_snapshot.items():
                    if coord in pattern_cells:
                        continue
                    if after_snap.get(coord) != before_val:
                        changed_outside = True
                        break
                lines.append("reward result: requested")
                lines.append(f"reward effect on this board: {'changed' if changed_outside else 'no_visible_change'}")
                lines.append("note: reward may still be impossible or may affect opponent board only.")
            else:
                # apply=None обычно означает, что движок не спрашивал агента (авто-применение)
                lines.append("reward result: engine_auto_or_not_requested")

        lines.append("board after clear:")
        lines.extend(_render_board(ps.board, highlights=None, mark="+"))

        self.ui.print_block(lines, pause_after=True)
        self.pending_pattern = None

    def _flush_turn(self, state: GameState, ctx: TurnContext) -> None:
        ps = state.players[ctx.player_idx]

        lines: List[str] = []
        lines.append("")
        lines.append(f"TURN {ctx.turn_number} END | Player {ctx.player_idx} ({ctx.agent_name})")
        lines.append(
            f"start: deck={ctx.start_deck} discard={ctx.start_discard} vp={ctx.start_vps} "
            f"hand={ctx.start_hand} incoming={ctx.start_incoming}"
        )

        if ctx.draft.revealed:
            d = ctx.draft
            reveal_str = ", ".join(f"[{i}] {t}" for i, t in enumerate(d.revealed))
            lines.append(f"draft: reveal {reveal_str}")

            pick = d.pick or "?"
            passed = d.passed or "?"
            discarded = d.discarded or "?"
            pass_to = d.pass_to_idx
            if pass_to is None:
                lines.append(f"draft result: pick={pick} pass=? discard={discarded}")
            else:
                lines.append(f"draft result: pick={pick} pass->{pass_to} {passed} discard={discarded}")

        if ctx.placements:
            lines.append("placements:")
            for pl in ctx.placements:
                lines.append(f" - {pl.card} -> {pl.cell}")
        else:
            lines.append("placements: none")

        lines.append("board at end of turn:")
        hl_cells = [pl.cell for pl in ctx.placements]
        lines.extend(_render_board(ps.board, highlights=hl_cells, mark="+"))

        self.ui.print_block(lines, pause_after=True)

    def ensure_turn(self, state: GameState) -> None:
        # если был паттерн на прошлом шаге, к этому моменту движок уже очистил клетки
        self._flush_pending_pattern_if_ready(state)

        key = (state.turn_number, state.current_player_idx)
        if self.current_key == key:
            return

        # новый ход: сначала выводим предыдущий ход (если был)
        if self.current_ctx is not None:
            self._flush_turn(state, self.current_ctx)

        pidx = state.current_player_idx
        agent_name = self.agents_by_player[pidx] if 0 <= pidx < len(self.agents_by_player) else "?"
        start_hand_tokens = [_card_token(c) for c in state.current_player().hand]
        incoming = self.incoming_by_player.get(pidx)

        self.current_key = key
        self.current_ctx = TurnContext(
            turn_number=state.turn_number,
            player_idx=pidx,
            agent_name=agent_name,
            start_deck=len(state.deck),
            start_discard=len(state.discard),
            start_hand=start_hand_tokens,
            start_incoming=incoming,
            start_vps=[p.vp for p in state.players],
        )

    def finish(self, final_state: GameState) -> None:
        # в конце игры тоже может висеть pending pattern и незасфлашенный последний ход
        self._flush_pending_pattern_if_ready(final_state)
        if self.current_ctx is not None:
            self._flush_turn(final_state, self.current_ctx)
            self.current_ctx = None
            self.current_key = None

    def on_draft_pick(self, state: GameState, revealed: Sequence[Card], pick_idx: int) -> None:
        self.ensure_turn(state)
        if self.current_ctx is None:
            return
        d = self.current_ctx.draft
        d.revealed = [_card_token(c) for c in revealed]
        if 0 <= pick_idx < len(revealed):
            d.pick = _card_token(revealed[pick_idx])

    def on_draft_pass(self, state: GameState, remaining: Sequence[Card], pass_idx: int) -> None:
        self.ensure_turn(state)
        if self.current_ctx is None:
            return
        d = self.current_ctx.draft

        nxt = state.player_left_of(state.current_player_idx)
        d.pass_to_idx = nxt

        if 0 <= pass_idx < len(remaining):
            d.passed = _card_token(remaining[pass_idx])

        if len(remaining) == 2:
            other = 1 - pass_idx
            if 0 <= other < 2:
                d.discarded = _card_token(remaining[other])

        # incoming отображаем в начале хода получателя
        if d.passed is not None:
            self.incoming_by_player[nxt] = d.passed

    def on_placement(self, state: GameState, card: Card, cell: Coord) -> None:
        self.ensure_turn(state)
        if self.current_ctx is None:
            return

        tok = _card_token(card)
        self.current_ctx.placements.append(TurnPlacement(card=tok, cell=cell))

        # если это была incoming-карта, убираем её из "incoming" следующего отображения
        pidx = state.current_player_idx
        inc = self.incoming_by_player.get(pidx)
        if inc == tok:
            self.incoming_by_player.pop(pidx, None)

    def on_pattern_chosen(self, state: GameState, chosen: Any) -> None:
        self.ensure_turn(state)

        pidx = state.current_player_idx
        ps = state.players[pidx]

        cells_raw = list(getattr(chosen, "cells"))
        cells: List[Coord] = [(int(x), int(y)) for (x, y) in cells_raw]

        before_lines = _render_board(ps.board, highlights=cells, mark="*")
        before_snapshot = _snapshot_board_tokens(ps.board)

        self.pending_pattern = PendingPattern(
            player_idx=pidx,
            turn_number=state.turn_number,
            pattern_id=str(getattr(chosen, "pattern_id")),
            vp=int(getattr(chosen, "vp")),
            reward_id=getattr(chosen, "reward_id", None),
            cells=cells,
            board_before_lines=before_lines,
            before_snapshot=before_snapshot,
        )

    def on_reward_apply(self, apply: bool) -> None:
        if self.pending_pattern is None:
            return
        self.pending_pattern.apply_reward = apply

    def on_reward_params(self, params: Dict[str, Any]) -> None:
        if self.pending_pattern is None:
            return
        self.pending_pattern.reward_params = params


class WatchingAgent(Agent):
    def __init__(self, inner: Agent, logger: TurnLogger) -> None:
        self._inner = inner
        self._logger = logger

    @property
    def name(self) -> str:
        return self._inner.name

    def choose_placement_cell(self, state: GameState, card: Card) -> Coord:
        self._logger.ensure_turn(state)
        cell = self._inner.choose_placement_cell(state, card)
        self._logger.on_placement(state, card, cell)
        return cell

    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        self._logger.ensure_turn(state)
        idx = self._inner.choose_draft_pick(state, revealed_cards)
        self._logger.on_draft_pick(state, revealed_cards, idx)
        return idx

    def choose_draft_pass(self, state: GameState, remaining_cards: Sequence[Card]) -> int:
        self._logger.ensure_turn(state)
        idx = self._inner.choose_draft_pass(state, remaining_cards)
        self._logger.on_draft_pass(state, remaining_cards, idx)
        return idx

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[Any]) -> int:
        self._logger.ensure_turn(state)
        idx = self._inner.choose_pattern_to_resolve(state, found_patterns)
        if 0 <= idx < len(found_patterns):
            self._logger.on_pattern_chosen(state, found_patterns[idx])
        return idx

    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        self._logger.ensure_turn(state)
        ans = self._inner.choose_apply_reward(state, reward_id, context)
        self._logger.on_reward_apply(ans)
        return ans

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        self._logger.ensure_turn(state)
        params = self._inner.choose_reward_params(state, reward_id, context)
        self._logger.on_reward_params(params)
        return params


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="interactive-watch")
    p.add_argument("--config", default="config/game_rules_config_v0_1.json")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--agents", nargs="+", default=["center_line4", "center_line4"])
    p.add_argument("--no-pause", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cfg_path = _resolve_config_path(args.config)
    cfg = load_rules_config(cfg_path)
    rewards_by_id = build_rewards_by_id(cfg)

    if args.players < 2:
        raise ValueError("--players must be >= 2")

    if len(args.agents) == 1:
        agent_names = [args.agents[0] for _ in range(args.players)]
    elif len(args.agents) == args.players:
        agent_names = list(args.agents)
    else:
        raise ValueError("Provide 1 agent name for all players, or exactly N names in --agents.")

    base_rng = random.Random(args.seed)
    deck_rng = random.Random(base_rng.getrandbits(64))
    engine_rng = random.Random(base_rng.getrandbits(64))

    # важное исправление: create_initial_game_state ожидает num_players и rng
    state = create_initial_game_state(cfg, num_players=args.players, rng=deck_rng, game_id=1)

    ui = WatchUI(pause_each_turn=(not args.no_pause))
    logger = TurnLogger(ui, agents_by_player=agent_names)

    agents: List[Agent] = []
    for i in range(args.players):
        cls = _import_agent_class(agent_names[i])
        arng = random.Random(base_rng.getrandbits(64))
        inner = _instantiate_agent(cls, rng=arng, rewards_by_id=rewards_by_id, cfg=cfg, player_idx=i)
        agents.append(WatchingAgent(inner, logger))

    engine = _instantiate_game_engine(cfg, agents, engine_rng)

    print("Interactive watch started.")
    print("Controls: Enter to continue, q + Enter to quit, Ctrl+C to quit.")

    # первая пауза перед началом
    ui.pause()

    final_state, stats, _events = engine.play_game(state)

    # добить последний ход и возможный pending pattern
    logger.finish(final_state)

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
