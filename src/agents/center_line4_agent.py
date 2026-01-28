from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.base import Agent, RewardContext
from config_loader import PatternSpec, RulesConfig
from models import Card, Coord, GameState, Joker, PatternMatch, PlacedCard, ProtectedZone
from pattern_engine import _Window, _match_color_rule, _match_rank_rule, _precompute_windows, find_patterns_on_board
from reward_engine import list_valid_reward_params


@dataclass(frozen=True, slots=True)
class _LineSlot:
    coords: Tuple[Coord, ...]
    protected_coords: Tuple[Coord, ...]
    outside_coord: Coord


class CenterLine4Agent(Agent):
    def __init__(self, rng: random.Random, cfg: RulesConfig) -> None:
        self._rng = rng
        self._cfg = cfg

        self._width = cfg.game.grid.width
        self._height = cfg.game.grid.height
        self._center_x = self._width // 2
        self._center_y = self._height // 2

        self._pz = ProtectedZone(cfg.game.grid.protected_zone.size)

        self._ranks_domain = tuple(cfg.game.deck.ranks)
        self._colors_domain = tuple(cfg.game.deck.colors)

        self._line4_patterns: List[PatternSpec] = [p for p in cfg.patterns if p.shape == "line_len_4"]
        if not self._line4_patterns:
            raise ValueError("No line_len_4 patterns in config")

        self._slots: List[_LineSlot] = self._build_center_slots()

        # Все line_len_4 окна для “внешнего режима”
        self._all_line4_windows: List[_Window] = _precompute_windows(self._width, self._height).get("line_len_4", [])

    @property
    def name(self) -> str:
        return "center_line4"

    def choose_placement_cell(self, state: GameState, card: Card) -> Coord:
        empties = state.current_player().board.empty_cells()
        if not empties:
            return (0, 0)

        best_score: Optional[float] = None
        best_cells: List[Coord] = []

        for cell in empties:
            s = self._score_placement(state, card, cell)
            if best_score is None or s > best_score:
                best_score = s
                best_cells = [cell]
            elif s == best_score:
                best_cells.append(cell)

        return self._rng.choice(best_cells)

    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        if not revealed_cards:
            return 0

        empties = state.current_player().board.empty_cells()
        if not empties:
            return 0

        best_i = 0
        best_score: Optional[float] = None
        for i, c in enumerate(revealed_cards):
            s = self._best_placement_score_for_card(state, c, empties)
            if best_score is None or s > best_score:
                best_score = s
                best_i = i
        return best_i

    def choose_draft_pass(self, state: GameState, remaining_cards: Sequence[Card]) -> int:
        if not remaining_cards:
            return 0
        if len(remaining_cards) == 1:
            return 0

        empties = state.current_player().board.empty_cells()
        if not empties:
            return 0

        scores = [self._best_placement_score_for_card(state, c, empties) for c in remaining_cards]

        # Передаём карту, которая по нашей оценке слабее (минимизируем помощь оппоненту),
        # а более сильная уйдёт в общий сброс.
        return 0 if scores[0] <= scores[1] else 1

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[PatternMatch]) -> int:
        if not found_patterns:
            return 0

        best_i = 0
        best_key: Optional[Tuple[int, int]] = None  # (has_rwd4, vp)

        for i, pm in enumerate(found_patterns):
            has_rwd4 = 1 if (pm.reward_id == "RWD4") else 0
            key = (has_rwd4, pm.vp)
            if best_key is None or key > best_key:
                best_key = key
                best_i = i

        return best_i

    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        valid = list_valid_reward_params(state, self._cfg, reward_id)
        if not valid:
            return False

        if reward_id in ("RWD4", "RWD5"):
            return True

        # RWD10 применять только если есть реально нужная карта
        if reward_id == "RWD10":
            return self._best_rwd10_benefit(state, valid) > 0.0

        if reward_id in ("RWD2", "RWD8", "RWD9"):
            return self._best_reward_placement_benefit(state, reward_id, valid) > 0.0

        if reward_id in ("RWD1", "RWD3"):
            empties = len(state.current_player().board.empty_cells())
            return empties <= 2

        if reward_id in ("RWD6", "RWD7"):
            me = state.current_player().vp
            lead = self._leader_opponent_vp(state)
            return lead is not None and lead > me

        return False

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        valid = list_valid_reward_params(state, self._cfg, reward_id)
        if not valid:
            return {}

        if reward_id in ("RWD2", "RWD8", "RWD9"):
            return self._choose_best_reward_placement_params(state, reward_id, valid) or valid[0]

        if reward_id == "RWD10":
            return self._choose_best_rwd10_params(state, valid) or valid[0]

        return self._rng.choice(valid)

    def _build_center_slots(self) -> List[_LineSlot]:
        windows = _precompute_windows(self._width, self._height).get("line_len_4", [])
        out: List[_LineSlot] = []
        for w in windows:
            if w.size != 4:
                continue
            coords = w.coords
            prot = [c for c in coords if self._is_protected(c[0], c[1])]
            if len(prot) != 3:
                continue
            outside = [c for c in coords if c not in prot]
            if len(outside) != 1:
                continue
            out.append(_LineSlot(coords=coords, protected_coords=tuple(prot), outside_coord=outside[0]))
        return out

    def _is_protected(self, x: int, y: int) -> bool:
        return self._pz.is_protected(x, y, self._width, self._height)

    def _protected_has_space(self, state: GameState) -> bool:
        b = state.current_player().board
        for y in range(self._height):
            for x in range(self._width):
                if self._is_protected(x, y) and b.get(x, y) is None:
                    return True
        return False

    def _best_placement_score_for_card(self, state: GameState, card: Card, empties: Sequence[Coord]) -> float:
        best = -1e18
        for cell in empties:
            s = self._score_placement(state, card, cell)
            if s > best:
                best = s
        return best

    def _score_placement(self, state: GameState, placed: PlacedCard, cell: Coord) -> float:
        board = state.current_player().board
        x, y = cell

        if not board.is_empty(x, y):
            return -1e18

        # Немедленный VP: доминирующий фактор
        immediate_best_vp = self._immediate_best_vp_after_placement(board, placed, cell)
        score = float(immediate_best_vp) * 1000.0

        # Включаем режимы:
        center_mode = self._protected_has_space(state)

        if center_mode:
            # Пока в protected есть места, мы давим на “центр-сейф”
            if self._is_protected(x, y):
                score += 40.0
            else:
                score -= 20.0

            # Продвижение по слотам 3 protected + 1 outside
            score += self._center_slot_progress_score(board, placed, cell, immediate_best_vp)

        else:
            # Центр заполнен: строим линии снаружи. Тут важна двусторонняя расширяемость
            # и старт ближе к центру ряда/колонки.
            score += self._outside_line_build_score(board, placed, cell)

        return score

    def _center_slot_progress_score(self, board, placed: PlacedCard, cell: Coord, immediate_best_vp: int) -> float:
        score = 0.0
        for slot in self._slots:
            if cell not in slot.coords:
                continue

            protected_filled_before = self._count_slot_protected_filled(board, slot, None, None)
            protected_filled_after = self._count_slot_protected_filled(board, slot, cell, placed)

            if cell in slot.protected_coords:
                score += float(protected_filled_after - protected_filled_before) * 12.0
                score += float(protected_filled_after) * 3.0

            if cell == slot.outside_coord:
                # outside клетку в центр-режиме стараемся ставить только на “добивку”
                if immediate_best_vp == 0:
                    score -= 60.0
                if protected_filled_before < 3:
                    score -= 25.0

            # Потенциал совместимости по правилам line_len_4 (с пустыми как Joker)
            compat_vp = self._compatibility_vp_for_coords(board, slot.coords, cell, placed)
            score += compat_vp * (0.5 + float(protected_filled_after) / 3.0)

        return score

    def _outside_line_build_score(self, board, placed: PlacedCard, cell: Coord) -> float:
        x, y = cell
        score = 0.0

        # Бонус за близость к центру доски (приближение к “центру ряда/колонки” в среднем)
        dist_center = abs(x - self._center_x) + abs(y - self._center_y)
        score += float((self._width + self._height) // 2 - dist_center) * 1.5

        # Совместимость с line_len_4 паттернами во всех окнах, которые содержат cell
        compat_sum = 0.0
        for w in self._all_line4_windows:
            if cell not in w.coords:
                continue
            compat_sum += self._compatibility_vp_for_coords(board, w.coords, cell, placed)
        score += compat_sum * 0.8

        # Двусторонняя расширяемость: хотим иметь возможность наращивать линию в обе стороны
        horiz = self._extendability_score(board, placed, cell, dx=1, dy=0) + self._row_center_bonus(x)
        vert = self._extendability_score(board, placed, cell, dx=0, dy=1) + self._col_center_bonus(y)
        score += max(horiz, vert) * 6.0

        return score

    def _row_center_bonus(self, x: int) -> float:
        # “начинать ближе к центру ряда”
        return float((self._width // 2) - abs(x - self._center_x)) * 2.0

    def _col_center_bonus(self, y: int) -> float:
        # “начинать ближе к центру колонки”
        return float((self._height // 2) - abs(y - self._center_y)) * 2.0

    def _extendability_score(self, board, placed: PlacedCard, cell: Coord, dx: int, dy: int) -> float:
        """
        Считает, насколько хорошо эта постановка поддерживает рост линии в обе стороны.
        Идея: чем больше непрерывная группа занятых клеток и чем больше “открытых концов”,
        тем лучше.
        """
        x, y = cell
        board.set(x, y, placed)
        try:
            run_len = 1
            open_ends = 0

            # Вперёд
            cx, cy = x + dx, y + dy
            while board.in_bounds(cx, cy) and board.get(cx, cy) is not None:
                run_len += 1
                cx, cy = cx + dx, cy + dy
            if board.in_bounds(cx, cy) and board.get(cx, cy) is None:
                open_ends += 1

            # Назад
            cx, cy = x - dx, y - dy
            while board.in_bounds(cx, cy) and board.get(cx, cy) is not None:
                run_len += 1
                cx, cy = cx - dx, cy - dy
            if board.in_bounds(cx, cy) and board.get(cx, cy) is None:
                open_ends += 1

            return float(run_len * run_len) + float(open_ends) * 2.5
        finally:
            board.set(x, y, None)

    def _compatibility_vp_for_coords(self, board, coords: Tuple[Coord, ...], placed_cell: Coord, placed_card: PlacedCard) -> float:
        """
        Берём окно coords длины 4, подставляем placed_card в placed_cell,
        все пустые клетки трактуем как Joker (потенциал), и суммируем VP всех
        line_len_4 паттернов, которые остаются совместимыми.
        """
        placed_cards: List[PlacedCard] = []
        for c in coords:
            if c == placed_cell:
                placed_cards.append(placed_card)
                continue
            v = board.get(c[0], c[1])
            placed_cards.append(v if v is not None else Joker())

        w = _Window(coords=coords, shape="line_len_4", size=4)

        compat_vp = 0.0
        for pat in self._line4_patterns:
            if _match_rank_rule(placed_cards, w, pat.rank_rule, self._ranks_domain) and _match_color_rule(
                placed_cards, w, pat.color_rule, self._colors_domain
            ):
                compat_vp += float(pat.vp)
        return compat_vp

    def _immediate_best_vp_after_placement(self, board, placed: PlacedCard, cell: Coord) -> int:
        x, y = cell
        if not board.is_empty(x, y):
            return 0
        board.set(x, y, placed)
        try:
            pats = find_patterns_on_board(board, self._cfg)
            if not pats:
                return 0
            return max(pm.vp for pm in pats)
        finally:
            board.set(x, y, None)

    def _count_slot_protected_filled(self, board, slot: _LineSlot, placed_cell: Optional[Coord], placed_card: Optional[PlacedCard]) -> int:
        cnt = 0
        for (x, y) in slot.protected_coords:
            if placed_cell is not None and (x, y) == placed_cell:
                cnt += 1
                continue
            if board.get(x, y) is not None:
                cnt += 1
        return cnt

    def _leader_opponent_vp(self, state: GameState) -> Optional[int]:
        me_idx = state.current_player_idx
        best: Optional[int] = None
        for i, p in enumerate(state.players):
            if i == me_idx:
                continue
            if best is None or p.vp > best:
                best = p.vp
        return best

    def _best_reward_placement_benefit(self, state: GameState, reward_id: str, valid_params: Sequence[Dict[str, Any]]) -> float:
        best = -1e18
        for params in valid_params:
            placed = self._placed_card_for_reward_preview(state, reward_id, params)
            if placed is None:
                continue
            dst = params.get("self_empty_coord")
            if not isinstance(dst, tuple) or len(dst) != 2:
                continue
            s = self._score_placement(state, placed, (int(dst[0]), int(dst[1])))
            if s > best:
                best = s
        if best <= -1e17:
            return 0.0
        return best

    def _choose_best_reward_placement_params(self, state: GameState, reward_id: str, valid_params: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best_params: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None

        for params in valid_params:
            placed = self._placed_card_for_reward_preview(state, reward_id, params)
            if placed is None:
                continue
            dst = params.get("self_empty_coord")
            if not isinstance(dst, tuple) or len(dst) != 2:
                continue
            s = self._score_placement(state, placed, (int(dst[0]), int(dst[1])))
            if best_score is None or s > best_score:
                best_score = s
                best_params = params

        return best_params

    def _placed_card_for_reward_preview(self, state: GameState, reward_id: str, params: Dict[str, Any]) -> Optional[PlacedCard]:
        if reward_id == "RWD9":
            return Joker()

        if reward_id == "RWD2":
            if len(state.discard) == 0:
                return None
            return state.discard[-1]

        if reward_id == "RWD8":
            if len(state.discard) == 0:
                return None
            idx_from_top = params.get("discard_index_from_top")
            if not isinstance(idx_from_top, int):
                return None
            if idx_from_top < 0 or idx_from_top >= len(state.discard):
                return None
            real_index = len(state.discard) - 1 - idx_from_top
            return state.discard[real_index]

        return None

    def _best_rwd10_benefit(self, state: GameState, valid_params: Sequence[Dict[str, Any]]) -> float:
        best = -1e18
        for params in valid_params:
            placed = self._stolen_card_preview(state, params)
            if placed is None:
                continue
            dst = params.get("self_empty_coord")
            if not isinstance(dst, tuple) or len(dst) != 2:
                continue
            s = self._score_placement(state, placed, (int(dst[0]), int(dst[1])))
            if s > best:
                best = s
        if best <= -1e17:
            return 0.0
        return best

    def _choose_best_rwd10_params(self, state: GameState, valid_params: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best_params: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None

        for params in valid_params:
            placed = self._stolen_card_preview(state, params)
            if placed is None:
                continue
            dst = params.get("self_empty_coord")
            if not isinstance(dst, tuple) or len(dst) != 2:
                continue
            s = self._score_placement(state, placed, (int(dst[0]), int(dst[1])))
            if best_score is None or s > best_score:
                best_score = s
                best_params = params

        return best_params

    def _stolen_card_preview(self, state: GameState, params: Dict[str, Any]) -> Optional[PlacedCard]:
        opp_idx = params.get("opponent_idx")
        src = params.get("opponent_card_coord")
        if not isinstance(opp_idx, int):
            return None
        if not isinstance(src, tuple) or len(src) != 2:
            return None
        ox, oy = int(src[0]), int(src[1])
        if opp_idx < 0 or opp_idx >= len(state.players):
            return None
        return state.players[opp_idx].board.get(ox, oy)
