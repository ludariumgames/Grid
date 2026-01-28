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
    """
    Линия длины 4, где ровно 3 клетки находятся в protected center (3x3),
    и ровно 1 клетка снаружи. Идея: заполнять protected часть, а наружную клетку
    ставить только на финише (когда паттерн сразу срабатывает и очищается).
    """
    coords: Tuple[Coord, ...]
    protected_coords: Tuple[Coord, ...]
    outside_coord: Coord


class CenterLine4Agent(Agent):
    def __init__(self, rng: random.Random, cfg: RulesConfig) -> None:
        self._rng = rng
        self._cfg = cfg

        self._pz = ProtectedZone(cfg.game.grid.protected_zone.size)
        self._width = cfg.game.grid.width
        self._height = cfg.game.grid.height

        self._ranks_domain = tuple(cfg.game.deck.ranks)
        self._colors_domain = tuple(cfg.game.deck.colors)

        self._line4_patterns: List[PatternSpec] = [p for p in cfg.patterns if p.shape == "line_len_4"]
        if not self._line4_patterns:
            raise ValueError("No line_len_4 patterns in config")

        self._slots: List[_LineSlot] = self._build_slots()

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
            s = self._score_placement_card(state, card, cell)
            if best_score is None or s > best_score:
                best_score = s
                best_cells = [cell]
            elif s == best_score:
                best_cells.append(cell)

        if len(best_cells) == 1:
            return best_cells[0]
        return self._rng.choice(best_cells)

    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        if not revealed_cards:
            return 0

        best_i = 0
        best_score: Optional[float] = None

        empties = state.current_player().board.empty_cells()
        if not empties:
            return 0

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

        # Мы выбираем карту, которую передадим, а оставшуюся движок выбросит в общий сброс.
        # Чтобы минимизировать пользу для оппонента, логично выбросить "самую полезную" (по нашей оценке),
        # а передать "самую бесполезную".
        pass_idx = 0 if scores[0] <= scores[1] else 1
        return pass_idx

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[PatternMatch]) -> int:
        if not found_patterns:
            return 0

        # Приоритет:
        # 1) паттерны с RWD4 (extra_turn)
        # 2) иначе максимальный VP
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

        # Темповые награды почти всегда хороши.
        if reward_id in ("RWD4", "RWD5"):
            return True

        # Украсть карту имеет смысл только если она реально помогает нашему плану (ты это уточнил).
        if reward_id == "RWD10":
            return self._best_rwd10_benefit(state, valid) > 0.0

        # Поставить джокер обычно полезно, но всё равно проверим, что это улучшает наш план.
        if reward_id == "RWD9":
            return self._best_reward_placement_benefit(state, reward_id, valid) > 0.0

        # Взять из сброса стоит только если карта улучшает план.
        if reward_id in ("RWD2", "RWD8"):
            return self._best_reward_placement_benefit(state, reward_id, valid) > 0.0

        # Удалить свою карту: применять только если очень мало места (страховка от END по месту).
        if reward_id == "RWD1":
            empties = len(state.current_player().board.empty_cells())
            return empties <= 2

        # Переместить свою: тоже только как попытка "разрулить" почти полный борд.
        if reward_id == "RWD3":
            empties = len(state.current_player().board.empty_cells())
            return empties <= 2

        # Воздействие на оппонента: применяем если мы отстаем по VP и есть валидные цели.
        if reward_id in ("RWD6", "RWD7"):
            me = state.current_player().vp
            leader = self._leader_opponent_vp(state)
            return leader is not None and leader > me

        return False

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        valid = list_valid_reward_params(state, self._cfg, reward_id)
        if not valid:
            return {}

        # Для placement-наград выбираем параметр с максимальной выгодой по нашему плану.
        if reward_id in ("RWD2", "RWD8", "RWD9"):
            return self._choose_best_reward_placement_params(state, reward_id, valid) or valid[0]

        if reward_id == "RWD10":
            return self._choose_best_rwd10_params(state, valid) or valid[0]

        # Для остальных пока берём случайный валидный (движок всё равно валидирует).
        return self._rng.choice(valid)

    def _build_slots(self) -> List[_LineSlot]:
        windows = _precompute_windows(self._width, self._height).get("line_len_4", [])
        out: List[_LineSlot] = []
        for w in windows:
            if w.size != 4:
                continue
            coords = w.coords
            prot = [c for c in coords if self._pz.is_protected(c[0], c[1], self._width, self._height)]
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
            s = self._score_placement_card(state, card, cell)
            if s > best:
                best = s
        return best

    def _score_placement_card(self, state: GameState, card: Card, cell: Coord) -> float:
        return self._score_placement_any(state, card, cell)

    def _score_placement_any(self, state: GameState, placed: PlacedCard, cell: Coord) -> float:
        """
        Скоринг для размещения (Card или Joker) в cell на поле текущего игрока.
        Мы сильно предпочитаем:
        - немедленный триггер паттерна
        - заполнение protected части line4-слотов
        И сильно штрафуем раннюю установку "outside" клетки, если паттерн не срабатывает.
        """
        board = state.current_player().board
        x, y = cell

        # Немедленный VP: ставим карту, ищем паттерны, откатываем.
        immediate_best_vp = self._immediate_best_vp_after_placement(board, placed, cell)

        # Базовый бонус за protected
        score = 0.0
        if self._is_protected(x, y):
            score += 0.5

        # Немедленный VP должен доминировать выбор.
        score += float(immediate_best_vp) * 1000.0

        # Потенциал line4: сколько слотов мы продвигаем к "3 protected + финиш".
        for slot in self._slots:
            if cell not in slot.coords:
                continue

            protected_filled_before = self._count_slot_protected_filled(board, slot, placed_cell=None, placed_card=None)
            protected_filled_after = self._count_slot_protected_filled(board, slot, placed_cell=cell, placed_card=placed)

            # Если мы заполняем protected часть, это хорошо.
            if cell in slot.protected_coords:
                score += float(protected_filled_after - protected_filled_before) * 10.0
                score += float(protected_filled_after) * 2.0

            # Если мы ставим outside слишком рано, это плохо.
            if cell == slot.outside_coord:
                # Если паттерн не сработал сразу, мы оставили уязвимую клетку.
                if immediate_best_vp == 0:
                    score -= 50.0

                # Даже если protected еще не собран полностью, outside обычно не хотим.
                if protected_filled_before < 3:
                    score -= 20.0

            # Совместимость с line_len_4 паттернами (H/I/J) при незаполненных клетках.
            compat_vp = self._compatibility_vp_line4_slot(board, slot, placed_cell=cell, placed_card=placed)
            # Чем ближе protected часть к 3, тем выше ценность совместимости.
            score += compat_vp * (0.5 + (float(protected_filled_after) / 3.0))

        return score

    def _immediate_best_vp_after_placement(self, board, placed: PlacedCard, cell: Coord) -> int:
        x, y = cell
        if not board.is_empty(x, y):
            return 0
        board.set(x, y, placed)
        try:
            pats = find_patterns_on_board(board, self._cfg)
            if not pats:
                return 0
            # По твоей логике: одна карта обычно приводит максимум к одному реальному паттерну.
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

    def _compatibility_vp_line4_slot(self, board, slot: _LineSlot, placed_cell: Coord, placed_card: PlacedCard) -> float:
        """
        Рассматриваем слот как частично заполненный: пустые клетки трактуем как Joker (неизвестное значение),
        и проверяем, остаётся ли линия совместимой с правилами rank_rule/color_rule паттернов line_len_4.
        """
        placed_cards: List[PlacedCard] = []
        for coord in slot.coords:
            if coord == placed_cell:
                placed_cards.append(placed_card)
                continue
            c = board.get(coord[0], coord[1])
            placed_cards.append(c if c is not None else Joker())

        w = _Window(coords=slot.coords, shape="line_len_4", size=4)

        compat_vp = 0.0
        for pat in self._line4_patterns:
            if _match_rank_rule(placed_cards, w, pat.rank_rule, self._ranks_domain) and _match_color_rule(
                placed_cards, w, pat.color_rule, self._colors_domain
            ):
                compat_vp += float(pat.vp)
        return compat_vp

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
        """
        Возвращает максимальную выгоду (по нашему скорингу) среди валидных вариантов.
        Выгода считается относительно текущего состояния, через "score placement" в выбранной клетке.
        """
        base = 0.0
        best = -1e18
        for params in valid_params:
            placed = self._placed_card_for_reward_preview(state, reward_id, params)
            if placed is None:
                continue
            dst = params.get("self_empty_coord")
            if not isinstance(dst, tuple) or len(dst) != 2:
                continue
            s = self._score_placement_any(state, placed, (int(dst[0]), int(dst[1])))
            if s > best:
                best = s
        if best <= -1e17:
            return 0.0
        return best - base

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
            cell = (int(dst[0]), int(dst[1]))
            s = self._score_placement_any(state, placed, cell)
            if best_score is None or s > best_score:
                best_score = s
                best_params = params

        return best_params

    def _placed_card_for_reward_preview(self, state: GameState, reward_id: str, params: Dict[str, Any]) -> Optional[PlacedCard]:
        """
        Возвращает карту, которую награда положит на наше поле, не меняя state.
        """
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
        """
        RWD10: применять только если есть карта, которая нам нужна.
        Мы считаем "нужна", если после кражи и размещения в dst скоринг улучшается.
        """
        best = -1e18
        for params in valid_params:
            placed = self._stolen_card_preview(state, params)
            if placed is None:
                continue
            dst = params.get("self_empty_coord")
            if not isinstance(dst, tuple) or len(dst) != 2:
                continue
            s = self._score_placement_any(state, placed, (int(dst[0]), int(dst[1])))
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
            cell = (int(dst[0]), int(dst[1]))
            s = self._score_placement_any(state, placed, cell)
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
