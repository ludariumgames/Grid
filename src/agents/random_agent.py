from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.base import Agent, RewardContext
from config_loader import RewardSpec
from models import Card, Coord, GameState, PatternMatch


class RandomAgent(Agent):
    def __init__(self, rng: random.Random, rewards_by_id: Dict[str, RewardSpec]) -> None:
        self._rng = rng
        self._rewards_by_id = rewards_by_id

    @property
    def name(self) -> str:
        return "random"

    def choose_placement_cell(self, state: GameState, card: Card) -> Coord:
        empties = state.current_player().board.empty_cells()
        if not empties:
            return (0, 0)
        return self._rng.choice(empties)

    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        if not revealed_cards:
            return 0
        return self._rng.randrange(len(revealed_cards))

    def choose_draft_pass(self, state: GameState, remaining_cards: Sequence[Card]) -> int:
        if not remaining_cards:
            return 0
        return self._rng.randrange(len(remaining_cards))

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[PatternMatch]) -> int:
        if not found_patterns:
            return 0
        return self._rng.randrange(len(found_patterns))

    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        spec = self._rewards_by_id.get(reward_id)
        if spec is None:
            return False

        if not self._is_reward_possible(state, spec):
            return False

        return bool(self._rng.getrandbits(1))

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        spec = self._rewards_by_id.get(reward_id)
        if spec is None:
            return {}

        try:
            return self._build_params_from_agent_choices(state, spec)
        except Exception:
            return {}

    def _is_reward_possible(self, state: GameState, spec: RewardSpec) -> bool:
        pid = state.current_player_idx
        me = state.players[pid]
        me_board = me.board

        if spec.constraints.requires_free_cell_on_self_board and not me_board.has_space():
            return False

        if spec.id in ("RWD2", "RWD8") and len(state.discard) == 0:
            return False

        if spec.id == "RWD1":
            return len(me_board.occupied_cells()) > 0

        if spec.id == "RWD3":
            return len(me_board.occupied_cells()) > 0 and len(me_board.empty_cells()) > 0

        if spec.id == "RWD4":
            return True

        if spec.id == "RWD5":
            return True

        if spec.id == "RWD6":
            return self._has_any_valid_opponent_card(state, forbid_protected=spec.constraints.forbid_opponent_protected_zone_as_source)

        if spec.id == "RWD7":
            # Нужно: оппонентская карта (src) и оппонентская пустая клетка (dst) с учетом запретов
            for opp in self._opponent_indices(state):
                if self._opponent_cards(state, opp, forbid_protected=spec.constraints.forbid_opponent_protected_zone_as_source):
                    if self._opponent_empty_cells(state, opp, forbid_protected=spec.constraints.forbid_opponent_protected_zone_as_destination):
                        return True
            return False

        if spec.id == "RWD9":
            return len(me_board.empty_cells()) > 0

        if spec.id == "RWD10":
            if len(me_board.empty_cells()) == 0:
                return False
            return self._has_any_valid_opponent_card(state, forbid_protected=spec.constraints.forbid_opponent_protected_zone_as_source)

        return True

    def _build_params_from_agent_choices(self, state: GameState, spec: RewardSpec) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        for choice in spec.agent_choices:
            if choice == "choose_self_card_coord":
                coord = self._choose_self_card_coord(state)
                if coord is None:
                    return {}
                params["self_card_coord"] = coord

            elif choice == "choose_self_empty_cell_coord":
                coord = self._choose_self_empty_coord(state)
                if coord is None:
                    return {}
                params["self_empty_coord"] = coord

            elif choice == "choose_opponent_id":
                opp = self._choose_opponent_id(state)
                if opp is None:
                    return {}
                params["opponent_idx"] = opp

            elif choice == "choose_opponent_card_coord":
                opp = params.get("opponent_idx")
                if opp is None:
                    return {}
                coord = self._choose_opponent_card_coord(
                    state,
                    opponent_idx=int(opp),
                    forbid_protected=spec.constraints.forbid_opponent_protected_zone_as_source,
                )
                if coord is None:
                    return {}
                params["opponent_card_coord"] = coord

            elif choice == "choose_opponent_empty_cell_coord":
                opp = params.get("opponent_idx")
                if opp is None:
                    return {}
                coord = self._choose_opponent_empty_coord(
                    state,
                    opponent_idx=int(opp),
                    forbid_protected=spec.constraints.forbid_opponent_protected_zone_as_destination,
                )
                if coord is None:
                    return {}
                params["opponent_empty_coord"] = coord

            elif choice == "choose_discard_card_ref":
                idx = self._choose_discard_index_from_top(state)
                if idx is None:
                    return {}
                params["discard_index_from_top"] = idx

            else:
                return {}

        return params

    def _opponent_indices(self, state: GameState) -> List[int]:
        n = len(state.players)
        me = state.current_player_idx
        return [i for i in range(n) if i != me]

    def _choose_opponent_id(self, state: GameState) -> Optional[int]:
        ops = self._opponent_indices(state)
        if not ops:
            return None
        return self._rng.choice(ops)

    def _choose_self_card_coord(self, state: GameState) -> Optional[Coord]:
        occ = state.current_player().board.occupied_cells()
        if not occ:
            return None
        coord, _ = self._rng.choice(occ)
        return coord

    def _choose_self_empty_coord(self, state: GameState) -> Optional[Coord]:
        empties = state.current_player().board.empty_cells()
        if not empties:
            return None
        return self._rng.choice(empties)

    def _opponent_cards(self, state: GameState, opponent_idx: int, forbid_protected: bool) -> List[Coord]:
        opp_board = state.players[opponent_idx].board
        out: List[Coord] = []
        for (x, y), _ in opp_board.occupied_cells():
            if forbid_protected and state.is_opponent_cell_protected(opponent_idx, x, y):
                continue
            out.append((x, y))
        return out

    def _opponent_empty_cells(self, state: GameState, opponent_idx: int, forbid_protected: bool) -> List[Coord]:
        opp_board = state.players[opponent_idx].board
        out: List[Coord] = []
        for (x, y) in opp_board.empty_cells():
            if forbid_protected and state.is_opponent_cell_protected(opponent_idx, x, y):
                continue
            out.append((x, y))
        return out

    def _choose_opponent_card_coord(self, state: GameState, opponent_idx: int, forbid_protected: bool) -> Optional[Coord]:
        coords = self._opponent_cards(state, opponent_idx, forbid_protected=forbid_protected)
        if not coords:
            return None
        return self._rng.choice(coords)

    def _choose_opponent_empty_coord(self, state: GameState, opponent_idx: int, forbid_protected: bool) -> Optional[Coord]:
        coords = self._opponent_empty_cells(state, opponent_idx, forbid_protected=forbid_protected)
        if not coords:
            return None
        return self._rng.choice(coords)

    def _choose_discard_index_from_top(self, state: GameState) -> Optional[int]:
        if len(state.discard) == 0:
            return None
        return self._rng.randrange(len(state.discard))

    def _has_any_valid_opponent_card(self, state: GameState, forbid_protected: bool) -> bool:
        for opp in self._opponent_indices(state):
            if self._opponent_cards(state, opp, forbid_protected=forbid_protected):
                return True
        return False
