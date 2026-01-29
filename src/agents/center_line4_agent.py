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


@dataclass(frozen=True, slots=True)
class _WindowMetric:
    pattern_id: str
    score: float
    filled_before: int
    filled_after: int
    protected_filled_after: int


class _BaseLine4Specialist(Agent):
    """
    Базовый агент для паттернов H/I/J (shape=line_len_4).

    Принципы:
    1) Продолжать уже начатые линии. Если нельзя - начинать новую линию в месте,
       где у линии будет "пространство" расти (лучше ближе к центру ряда/колонки).
    2) Protected (центр 3x3) используем только для "ценных" карт, то есть тех,
       что реально продвигают перспективную линию. Мусор туда не кладём.
    3) Мусорные карты кладём по краям, предпочтительно в углы, чтобы не блокировать.
    4) Для последовательностей (H/I) ранги 1 и 6 плохо стартуют, их чаще выгодно уводить
       на край, если они не продолжают линию.
    5) В драфте:
       - pick: максимизируем свой лучший placement-score
       - pass: минимизируем пользу следующему игроку (слева), анализируя ЕГО доску.
    """

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

        # Паттерны line_len_4 в конфиге
        self._line4_patterns: List[PatternSpec] = [p for p in cfg.patterns if p.shape == "line_len_4"]
        if not self._line4_patterns:
            raise ValueError("No line_len_4 patterns in config")

        self._patterns_by_id: Dict[str, PatternSpec] = {p.id: p for p in self._line4_patterns}

        # Все окна line_len_4
        self._all_line4_windows: List[_Window] = _precompute_windows(self._width, self._height).get("line_len_4", [])

        # "Центральные слоты" - это те line_len_4 окна, где ровно 3 клетки попадают в protected 3x3
        self._center_slots: List[_LineSlot] = self._build_center_slots()

        # Специализация конкретного наследника
        self._priority_pattern_ids: Tuple[str, ...] = self._build_priority()
        if not self._priority_pattern_ids:
            raise ValueError("Priority list must be non-empty")

        for pid in self._priority_pattern_ids:
            if pid not in self._patterns_by_id:
                raise ValueError(f"Pattern '{pid}' not found among line_len_4 patterns in config")

        self._primary_pattern_id: str = self._priority_pattern_ids[0]

        # Признак: основное правило ранга последовательность или все равны (J)
        self._primary_is_sequence = self._patterns_by_id[self._primary_pattern_id].rank_rule == "R1_line_sequence_step1"

    # ---- specialization hooks ----

    def _build_priority(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    # ---- Agent API ----

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

        nxt = state.player_left_of(state.current_player_idx)
        opp = state.players[nxt]
        opp_empties = opp.board.empty_cells()
        if not opp_empties:
            return 0

        # Чем меньше threat, тем лучше для нас (эту карту и передаём)
        threats = [self._opponent_threat_score(state, nxt, c, opp_empties) for c in remaining_cards]
        if threats[0] < threats[1]:
            return 0
        if threats[1] < threats[0]:
            return 1

        # tie-break: если одинаково, передаём карту с более высоким нашим placement-score (пусть лучше уйдёт в discard)
        my_empties = state.current_player().board.empty_cells()
        my_scores = [self._best_placement_score_for_card(state, c, my_empties) for c in remaining_cards]
        return 0 if my_scores[0] <= my_scores[1] else 1

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[PatternMatch]) -> int:
        if not found_patterns:
            return 0

        best_i = 0
        best_score: Optional[int] = None

        for i, pm in enumerate(found_patterns):
            s = self._pattern_pick_score(pm)
            if best_score is None or s > best_score:
                best_score = s
                best_i = i

        return best_i

    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        valid = list_valid_reward_params(state, self._cfg, reward_id)
        if not valid:
            return False

        # Родные награды паттернов H/I/J
        if reward_id == "RWD4":
            return True

        if reward_id == "RWD1":
            _best_params, benefit = self._best_rwd1_choice(state, valid)
            return benefit > 0.5

        if reward_id == "RWD7":
            _best_params, benefit = self._best_rwd7_choice(state, valid)
            return benefit > 0.5

        # Остальные награды: базовые евристики (чтобы не было "всё всегда отклонено")
        if reward_id == "RWD5":
            return True

        if reward_id == "RWD10":
            _best_params, benefit = self._best_rwd10_choice(state, valid)
            return benefit > 0.5

        if reward_id in ("RWD2", "RWD8", "RWD9"):
            _best_params, benefit = self._best_place_reward_choice(state, reward_id, valid)
            return benefit > 0.5

        if reward_id == "RWD3":
            _best_params, benefit = self._best_rwd3_choice(state, valid)
            return benefit > 0.5

        if reward_id == "RWD6":
            _best_params, benefit = self._best_rwd6_choice(state, valid)
            return benefit > 0.5

        return False

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        valid = list_valid_reward_params(state, self._cfg, reward_id)
        if not valid:
            return {}

        if reward_id == "RWD1":
            best_params, _benefit = self._best_rwd1_choice(state, valid)
            return best_params or valid[0]

        if reward_id == "RWD7":
            best_params, _benefit = self._best_rwd7_choice(state, valid)
            return best_params or valid[0]

        if reward_id == "RWD10":
            best_params, _benefit = self._best_rwd10_choice(state, valid)
            return best_params or valid[0]

        if reward_id in ("RWD2", "RWD8", "RWD9"):
            best_params, _benefit = self._best_place_reward_choice(state, reward_id, valid)
            return best_params or valid[0]

        if reward_id == "RWD3":
            best_params, _benefit = self._best_rwd3_choice(state, valid)
            return best_params or valid[0]

        if reward_id == "RWD6":
            best_params, _benefit = self._best_rwd6_choice(state, valid)
            return best_params or valid[0]

        return self._rng.choice(list(valid))

    # ---- scoring core ----

    def _score_placement(self, state: GameState, placed: PlacedCard, cell: Coord) -> float:
        board = state.current_player().board
        x, y = cell
        if not board.is_empty(x, y):
            return -1e18

        # 1) Немедленный результат: какой паттерн мы реально возьмём после постановки
        immediate = self._immediate_gain_after_placement(state, placed, cell)
        score = float(immediate.gained_vp) * 1000.0

        # Если сразу закрываем primary-паттерн, явно поощряем
        if immediate.chosen_pattern_id == self._primary_pattern_id and immediate.gained_vp > 0:
            score += 250.0

        # 2) Локальный прогресс по линиям длины 4 (наш приоритет)
        best_metrics = self._best_window_metrics_after_placement(board, placed, cell)
        if best_metrics:
            weight_base = 60.0
            for m in best_metrics:
                prio_idx = self._prio_index(m.pattern_id)
                if prio_idx is None:
                    continue
                w = weight_base / float(1 + prio_idx)
                score += m.score * w

        # 3) Ценность protected-зоны: защищаем только то, что реально участвует в перспективной линии
        is_prot = self._is_protected(x, y)
        valuable = self._is_valuable_placement(board, placed, cell, best_metrics, immediate.gained_vp)

        if is_prot:
            score += 35.0 if valuable else -80.0

        # 4) Мусорные карты по краям
        if not valuable:
            score += self._edge_corner_bonus(x, y) * 6.0

        # 5) Для последовательностей (H/I): ранги 1 и 6 хуже для старта
        if self._primary_is_sequence and isinstance(placed, Card):
            if placed.rank in (1, 6) and immediate.gained_vp == 0:
                if not valuable:
                    score += self._edge_corner_bonus(x, y) * 12.0
                if is_prot:
                    score -= 60.0

            if placed.rank in (3, 4) and immediate.gained_vp == 0 and valuable:
                score += self._center_proximity_bonus(x, y) * 3.0

        # 6) Слабый бонус: начинать ближе к центру ряда/колонки
        score += self._row_col_center_start_bonus(board, x, y) * 1.5

        return score

    # ---- immediate gain prediction ----

    @dataclass(frozen=True, slots=True)
    class _ImmediateGain:
        gained_vp: int
        chosen_pattern_id: Optional[str]
        chosen_reward_id: Optional[str]

    def _immediate_gain_after_placement(self, state: GameState, placed: PlacedCard, cell: Coord) -> _ImmediateGain:
        board = state.current_player().board
        x, y = cell
        board.set(x, y, placed)
        try:
            pats = find_patterns_on_board(board, self._cfg)
            if not pats:
                return self._ImmediateGain(0, None, None)

            best = max(pats, key=self._pattern_pick_score)
            return self._ImmediateGain(int(best.vp), str(best.pattern_id), getattr(best, "reward_id", None))
        finally:
            board.set(x, y, None)

    def _pattern_pick_score(self, pm: PatternMatch) -> int:
        prio = self._prio_index(pm.pattern_id)
        prio_score = 0 if prio is None else (100 - prio * 10)

        reward_bonus = 0
        if pm.reward_id == "RWD4":
            reward_bonus = 5
        elif pm.reward_id in ("RWD10", "RWD7", "RWD6"):
            reward_bonus = 3
        elif pm.reward_id in ("RWD2", "RWD8", "RWD9", "RWD1", "RWD3"):
            reward_bonus = 2

        return prio_score * 1000 + int(pm.vp) * 10 + reward_bonus

    def _prio_index(self, pattern_id: str) -> Optional[int]:
        try:
            return self._priority_pattern_ids.index(pattern_id)
        except ValueError:
            return None

    # ---- window metrics ----

    def _best_window_metrics_after_placement(self, board, placed: PlacedCard, cell: Coord) -> List[_WindowMetric]:
        out: List[_WindowMetric] = []

        x, y = cell
        board.set(x, y, placed)
        try:
            for pid in self._priority_pattern_ids:
                pat = self._patterns_by_id.get(pid)
                if pat is None:
                    continue

                best_m: Optional[_WindowMetric] = None
                for w in self._all_line4_windows:
                    if cell not in w.coords:
                        continue
                    m = self._window_metric_after_placement(board, w, pat)
                    if m is None:
                        continue
                    if best_m is None or m.score > best_m.score:
                        best_m = m

                if best_m is not None:
                    out.append(best_m)
        finally:
            board.set(x, y, None)

        return out

    def _window_metric_after_placement(self, board, w: _Window, pat: PatternSpec) -> Optional[_WindowMetric]:
        """
        board уже содержит "новую" карту в рассматриваемой клетке (которая находится в w.coords).
        Пустые клетки считаем как Joker (потенциал).
        """
        placed_cards: List[PlacedCard] = []
        filled_after = 0
        prot_filled_after = 0

        for (cx, cy) in w.coords:
            v = board.get(cx, cy)
            if v is None:
                placed_cards.append(Joker())
            else:
                placed_cards.append(v)
                filled_after += 1
                if self._is_protected(cx, cy):
                    prot_filled_after += 1

        if not _match_rank_rule(placed_cards, w, pat.rank_rule, self._ranks_domain):
            return None
        if not _match_color_rule(placed_cards, w, pat.color_rule, self._colors_domain):
            return None

        filled_before = max(0, filled_after - 1)

        # Метрика прогресса:
        # - продолжение важнее старта
        # - completion ценим отдельно
        completion_bonus = 60.0 if filled_after == 4 else 0.0
        score = float(filled_before * 14 + filled_after * 7) + float(prot_filled_after) * 3.0 + completion_bonus

        return _WindowMetric(
            pattern_id=pat.id,
            score=score,
            filled_before=filled_before,
            filled_after=filled_after,
            protected_filled_after=prot_filled_after,
        )

    def _window_potential_score(self, board, w: _Window, pat: PatternSpec) -> float:
        """
        Потенциал окна без допущений "мы только что поставили карту".
        """
        placed_cards: List[PlacedCard] = []
        filled = 0
        prot_filled = 0

        for (cx, cy) in w.coords:
            v = board.get(cx, cy)
            if v is None:
                placed_cards.append(Joker())
            else:
                placed_cards.append(v)
                filled += 1
                if self._is_protected(cx, cy):
                    prot_filled += 1

        if not _match_rank_rule(placed_cards, w, pat.rank_rule, self._ranks_domain):
            return 0.0
        if not _match_color_rule(placed_cards, w, pat.color_rule, self._colors_domain):
            return 0.0

        completion_bonus = 60.0 if filled == 4 else 0.0
        return float(filled * 18) + float(prot_filled) * 6.0 + completion_bonus

    def _is_valuable_placement(
        self,
        board,
        placed: PlacedCard,
        cell: Coord,
        metrics: List[_WindowMetric],
        immediate_vp: int,
    ) -> bool:
        if immediate_vp > 0:
            return True

        for m in metrics:
            if m.filled_after >= 2 and m.score >= 20.0:
                return True
            if m.filled_before >= 1 and m.score >= 16.0:
                return True

        return False

    # ---- opponent pass evaluation ----

    def _opponent_threat_score(self, state: GameState, opp_idx: int, card: Card, opp_empties: Sequence[Coord]) -> float:
        opp_board = state.players[opp_idx].board

        best_immediate_vp = 0
        best_potential = -1e18

        base_potential = self._board_line4_potential(opp_board)

        for cell in opp_empties:
            x, y = cell
            opp_board.set(x, y, card)
            try:
                pats = find_patterns_on_board(opp_board, self._cfg)
                if pats:
                    best_immediate_vp = max(best_immediate_vp, max(pm.vp for pm in pats))
                pot = self._board_line4_potential(opp_board)
                if pot > best_potential:
                    best_potential = pot
            finally:
                opp_board.set(x, y, None)

        pot_delta = 0.0 if best_potential <= -1e17 else (best_potential - base_potential)
        return float(best_immediate_vp) * 1000.0 + pot_delta * 10.0

    # ---- board potentials (for rewards) ----

    def _board_line4_potential(self, board) -> float:
        total = 0.0
        for w in self._all_line4_windows:
            for pid in self._priority_pattern_ids:
                pat = self._patterns_by_id.get(pid)
                if pat is None:
                    continue
                total += self._window_potential_score(board, w, pat)
        return total

    def _protected_blockers(self, board) -> int:
        blockers = 0
        for y in range(self._height):
            for x in range(self._width):
                if not self._is_protected(x, y):
                    continue
                if board.get(x, y) is None:
                    continue

                ok = False
                for w in self._all_line4_windows:
                    if (x, y) not in w.coords:
                        continue
                    for pid in self._priority_pattern_ids:
                        pat = self._patterns_by_id.get(pid)
                        if pat is None:
                            continue
                        if self._window_potential_score(board, w, pat) > 0.0:
                            ok = True
                            break
                    if ok:
                        break

                if not ok:
                    blockers += 1

        return blockers

    # ---- reward choices ----

    def _best_rwd1_choice(self, state: GameState, valid_params: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        me = state.current_player()
        b = me.board

        base_pot = self._board_line4_potential(b)
        base_blockers = self._protected_blockers(b)

        best_params: Optional[Dict[str, Any]] = None
        best_benefit = -1e18

        for params in valid_params:
            src = params.get("self_card_coord")
            if not isinstance(src, tuple) or len(src) != 2:
                continue
            sx, sy = int(src[0]), int(src[1])
            removed = b.get(sx, sy)
            if removed is None:
                continue

            b.set(sx, sy, None)
            try:
                pot = self._board_line4_potential(b)
                blockers = self._protected_blockers(b)
                benefit = (pot - base_pot) + float(base_blockers - blockers) * 25.0
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_params = params
            finally:
                b.set(sx, sy, removed)

        if best_benefit <= -1e17:
            return None, 0.0
        return best_params, float(best_benefit)

    def _best_rwd7_choice(self, state: GameState, valid_params: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        best_params: Optional[Dict[str, Any]] = None
        best_benefit = -1e18

        for params in valid_params:
            opp_idx = params.get("opponent_idx")
            src = params.get("opponent_card_coord")
            dst = params.get("opponent_empty_coord")
            if not isinstance(opp_idx, int):
                continue
            if not (isinstance(src, tuple) and len(src) == 2):
                continue
            if not (isinstance(dst, tuple) and len(dst) == 2):
                continue

            opp = state.players[opp_idx]
            ob = opp.board

            sx, sy = int(src[0]), int(src[1])
            dx, dy = int(dst[0]), int(dst[1])

            moving = ob.get(sx, sy)
            if moving is None:
                continue

            base_pot = self._board_line4_potential(ob)

            ob.set(sx, sy, None)
            ob.set(dx, dy, moving)
            try:
                pot = self._board_line4_potential(ob)
                benefit = base_pot - pot
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_params = params
            finally:
                ob.set(dx, dy, None)
                ob.set(sx, sy, moving)

        if best_benefit <= -1e17:
            return None, 0.0
        return best_params, float(best_benefit)

    def _best_rwd10_choice(self, state: GameState, valid_params: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        me = state.current_player()
        if not me.board.empty_cells():
            return None, 0.0

        best_params: Optional[Dict[str, Any]] = None
        best_score = -1e18

        for params in valid_params:
            opp_idx = params.get("opponent_idx")
            src = params.get("opponent_card_coord")
            dst = params.get("self_empty_coord")
            if not isinstance(opp_idx, int):
                continue
            if not isinstance(src, tuple) or len(src) != 2:
                continue
            if not isinstance(dst, tuple) or len(dst) != 2:
                continue

            ox, oy = int(src[0]), int(src[1])
            stolen = state.players[opp_idx].board.get(ox, oy)
            if stolen is None:
                continue

            cell = (int(dst[0]), int(dst[1]))
            s = self._score_placement(state, stolen, cell)
            if s > best_score:
                best_score = s
                best_params = params

        if best_score <= -1e17:
            return None, 0.0
        return best_params, float(best_score)

    def _best_place_reward_choice(self, state: GameState, reward_id: str, valid_params: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        best_params: Optional[Dict[str, Any]] = None
        best_score = -1e18

        for params in valid_params:
            placed = self._placed_card_for_reward_preview(state, reward_id, params)
            if placed is None:
                continue
            dst = params.get("self_empty_coord")
            if not isinstance(dst, tuple) or len(dst) != 2:
                continue
            cell = (int(dst[0]), int(dst[1]))
            s = self._score_placement(state, placed, cell)
            if s > best_score:
                best_score = s
                best_params = params

        if best_score <= -1e17:
            return None, 0.0
        return best_params, float(best_score)

    def _best_rwd3_choice(self, state: GameState, valid_params: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        me = state.current_player()
        b = me.board

        base_pot = self._board_line4_potential(b)

        best_params: Optional[Dict[str, Any]] = None
        best_benefit = -1e18

        for params in valid_params:
            src = params.get("self_card_coord")
            dst = params.get("self_empty_coord")
            if not (isinstance(src, tuple) and len(src) == 2):
                continue
            if not (isinstance(dst, tuple) and len(dst) == 2):
                continue
            sx, sy = int(src[0]), int(src[1])
            dx, dy = int(dst[0]), int(dst[1])

            moving = b.get(sx, sy)
            if moving is None:
                continue

            b.set(sx, sy, None)
            b.set(dx, dy, moving)
            try:
                pot = self._board_line4_potential(b)
                benefit = pot - base_pot
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_params = params
            finally:
                b.set(dx, dy, None)
                b.set(sx, sy, moving)

        if best_benefit <= -1e17:
            return None, 0.0
        return best_params, float(best_benefit)

    def _best_rwd6_choice(self, state: GameState, valid_params: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        best_params: Optional[Dict[str, Any]] = None
        best_benefit = -1e18

        for params in valid_params:
            opp_idx = params.get("opponent_idx")
            src = params.get("opponent_card_coord")
            if not isinstance(opp_idx, int):
                continue
            if not isinstance(src, tuple) or len(src) != 2:
                continue

            opp = state.players[opp_idx]
            ob = opp.board
            sx, sy = int(src[0]), int(src[1])

            removed = ob.get(sx, sy)
            if removed is None:
                continue

            base_pot = self._board_line4_potential(ob)

            ob.set(sx, sy, None)
            try:
                pot = self._board_line4_potential(ob)
                benefit = base_pot - pot
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_params = params
            finally:
                ob.set(sx, sy, removed)

        if best_benefit <= -1e17:
            return None, 0.0
        return best_params, float(best_benefit)

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

    # ---- utilities ----

    def _build_center_slots(self) -> List[_LineSlot]:
        out: List[_LineSlot] = []
        for w in self._all_line4_windows:
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

    def _best_placement_score_for_card(self, state: GameState, card: Card, empties: Sequence[Coord]) -> float:
        best = -1e18
        for cell in empties:
            s = self._score_placement(state, card, cell)
            if s > best:
                best = s
        return best

    def _edge_corner_bonus(self, x: int, y: int) -> float:
        is_corner = (x in (0, self._width - 1)) and (y in (0, self._height - 1))
        if is_corner:
            return 3.0
        is_edge = (x in (0, self._width - 1)) or (y in (0, self._height - 1))
        if is_edge:
            return 1.6
        return 0.0

    def _center_proximity_bonus(self, x: int, y: int) -> float:
        dist = abs(x - self._center_x) + abs(y - self._center_y)
        max_dist = (self._width - 1) + (self._height - 1)
        return float(max_dist - dist)

    def _row_col_center_start_bonus(self, board, x: int, y: int) -> float:
        if not board.is_empty(x, y):
            return 0.0
        row_bonus = float((self._width // 2) - abs(x - self._center_x))
        col_bonus = float((self._height // 2) - abs(y - self._center_y))
        return max(row_bonus, col_bonus)


class Line4HAgent(_BaseLine4Specialist):
    @property
    def name(self) -> str:
        return "line4_h"

    def _build_priority(self) -> Tuple[str, ...]:
        return ("H", "I", "J")


class Line4IAgent(_BaseLine4Specialist):
    @property
    def name(self) -> str:
        return "line4_i"

    def _build_priority(self) -> Tuple[str, ...]:
        return ("I", "H", "J")


class Line4JAgent(_BaseLine4Specialist):
    @property
    def name(self) -> str:
        return "line4_j"

    def _build_priority(self) -> Tuple[str, ...]:
        return ("J", "I", "H")


class CenterLine4Agent(Line4IAgent):
    """
    Алиас для существующего имени 'center_line4' (чтобы не ломать CLI/interactive_watch).
    По поведению это Line4IAgent (I->H->J), но name остаётся прежним.
    """

    @property
    def name(self) -> str:
        return "center_line4"
