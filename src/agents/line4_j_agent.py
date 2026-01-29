from __future__ import annotations

from typing import Sequence, Tuple

from agents.center_line4_agent import _BaseLine4Specialist
from models import Card, GameState


class Line4JAgent(_BaseLine4Specialist):
    """Специалист по паттерну J (line_len_4), с приоритетом J -> H -> I.

    Важно: choose_draft_pass оценивает карты по ДОСКЕ игрока слева и передает
    ту из двух оставшихся, которая дает ему меньший best_placement_score.
    """

    @property
    def name(self) -> str:
        return "line4_j"

    def _build_priority(self) -> Tuple[str, ...]:
        return ("J", "H", "I")

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

        # Считаем best_placement_score на доске игрока слева.
        # Никаких выводов "по своей доске" здесь не делаем.
        opp_view = self._state_view_for_player(state, nxt)
        s0 = self._best_placement_score_for_card(opp_view, remaining_cards[0], opp_empties)
        s1 = self._best_placement_score_for_card(opp_view, remaining_cards[1], opp_empties)

        # Передаем карту с меньшим best_placement_score.
        if s0 < s1:
            return 0
        if s1 < s0:
            return 1

        # Равенство: стабильный tie-break, передаем карту с меньшим immediate VP на доске соперника.
        i0 = self._best_immediate_vp_for_card(opp_view, remaining_cards[0], opp_empties)
        i1 = self._best_immediate_vp_for_card(opp_view, remaining_cards[1], opp_empties)
        if i0 < i1:
            return 0
        if i1 < i0:
            return 1

        return 0

    @staticmethod
    def _state_view_for_player(state: GameState, player_idx: int) -> GameState:
        # Создаем "view" на тот же state, но с другим current_player_idx.
        # Это не меняет исходный state и не трогает движок.
        return GameState(
            players=state.players,
            current_player_idx=player_idx,
            deck=state.deck,
            discard=state.discard,
            protected_zone=state.protected_zone,
            turn_number=state.turn_number,
            ended=state.ended,
            end_reason=state.end_reason,
            game_id=state.game_id,
        )

    def _best_immediate_vp_for_card(self, state_view: GameState, card: Card, empties) -> int:
        best = 0
        b = state_view.current_player().board
        for cell in empties:
            x, y = cell
            if not b.is_empty(x, y):
                continue
            g = self._immediate_gain_after_placement(state_view, card, cell)
            if g.gained_vp > best:
                best = int(g.gained_vp)
        return best
