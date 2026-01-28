from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config_loader import RewardSpec, RulesConfig
from models import Card, Coord, GameState, Joker, PlacedCard


class RewardError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RewardApplyResult:
    applied: bool
    impossible: bool
    extra_turn_granted: bool
    skip_turn_target_idx: Optional[int]
    notes: str


def build_rewards_by_id(cfg: RulesConfig) -> Dict[str, RewardSpec]:
    return {r.id: r for r in cfg.rewards}


def list_valid_reward_params(
    state: GameState,
    cfg: RulesConfig,
    reward_id: str,
    player_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Возвращает список валидных params для reward_id в текущем состоянии.
    Если список пустой, награда неприменима в данный момент.
    """
    rewards_by_id = build_rewards_by_id(cfg)
    spec = rewards_by_id.get(reward_id)
    if spec is None:
        raise RewardError(f"Unknown reward_id: {reward_id}")

    pid = state.current_player_idx if player_idx is None else player_idx
    me = state.players[pid]
    me_board = me.board

    if spec.constraints.requires_free_cell_on_self_board and not me_board.has_space():
        return []

    if reward_id == "RWD1":
        params: List[Dict[str, Any]] = []
        for coord, _ in me_board.occupied_cells():
            params.append({"self_card_coord": coord})
        return params

    if reward_id == "RWD2":
        if len(state.discard) == 0:
            return []
        params = [{"self_empty_coord": coord} for coord in me_board.empty_cells()]
        return params

    if reward_id == "RWD3":
        params: List[Dict[str, Any]] = []
        empties = me_board.empty_cells()
        if not empties:
            return []
        for coord, _ in me_board.occupied_cells():
            for dst in empties:
                if dst != coord:
                    params.append({"self_card_coord": coord, "self_empty_coord": dst})
        return params

    if reward_id == "RWD4":
        return [{}]

    if reward_id == "RWD5":
        return [{}]

    if reward_id == "RWD6":
        return _list_params_remove_opponent(state, cfg, spec, pid)

    if reward_id == "RWD7":
        return _list_params_move_opponent(state, cfg, spec, pid)

    if reward_id == "RWD8":
        if len(state.discard) == 0:
            return []
        empties = me_board.empty_cells()
        if not empties:
            return []
        params: List[Dict[str, Any]] = []
        for idx_from_top in range(len(state.discard)):
            for dst in empties:
                params.append({"discard_index_from_top": idx_from_top, "self_empty_coord": dst})
        return params

    if reward_id == "RWD9":
        return [{"self_empty_coord": coord} for coord in me_board.empty_cells()]

    if reward_id == "RWD10":
        if not me_board.has_space():
            return []
        return _list_params_steal_opponent(state, cfg, spec, pid)

    raise RewardError(f"Unsupported reward_id: {reward_id}")


def validate_reward_params(
    state: GameState,
    cfg: RulesConfig,
    reward_id: str,
    params: Dict[str, Any],
    player_idx: Optional[int] = None,
) -> bool:
    valid = list_valid_reward_params(state, cfg, reward_id, player_idx)
    return any(_params_equal(p, params) for p in valid)


def apply_reward(
    state: GameState,
    cfg: RulesConfig,
    reward_id: str,
    params: Dict[str, Any],
    player_idx: Optional[int] = None,
) -> RewardApplyResult:
    """
    Применяет награду. Если params невалидны, бросает RewardError.
    Если награда неприменима (например нет места/нет карт в сбросе), возвращает impossible=True.
    """
    pid = state.current_player_idx if player_idx is None else player_idx

    valid_params = list_valid_reward_params(state, cfg, reward_id, pid)
    if not valid_params:
        return RewardApplyResult(
            applied=False,
            impossible=True,
            extra_turn_granted=False,
            skip_turn_target_idx=None,
            notes="no_valid_params",
        )

    if not any(_params_equal(p, params) for p in valid_params):
        raise RewardError(f"Invalid params for {reward_id}: {params}")

    me = state.players[pid]
    me_board = me.board

    if reward_id == "RWD1":
        src = _require_coord(params, "self_card_coord")
        removed = me_board.remove(src[0], src[1])
        state.discard.append(removed)
        return RewardApplyResult(True, False, False, None, "removed_self_card")

    if reward_id == "RWD2":
        dst = _require_coord(params, "self_empty_coord")
        if len(state.discard) == 0:
            return RewardApplyResult(False, True, False, None, "discard_empty")
        taken = state.discard[-1]
        if not me_board.is_empty(dst[0], dst[1]):
            raise RewardError("Destination not empty")
        state.discard.pop()
        me_board.place(dst[0], dst[1], taken)
        return RewardApplyResult(True, False, False, None, "took_top_discard")

    if reward_id == "RWD3":
        src = _require_coord(params, "self_card_coord")
        dst = _require_coord(params, "self_empty_coord")
        me_board.move(src, dst)
        return RewardApplyResult(True, False, False, None, "moved_self_card")

    if reward_id == "RWD4":
        if cfg.termination.extra_turn_ignored_if_deck_empty and len(state.deck) == 0:
            return RewardApplyResult(True, False, False, None, "extra_turn_ignored_deck_empty")
        return RewardApplyResult(True, False, True, None, "extra_turn_granted")

    if reward_id == "RWD5":
        target = state.player_left_of(pid)
        state.players[target].skip_next_turn = True
        return RewardApplyResult(True, False, False, target, "skip_turn_set")

    if reward_id == "RWD6":
        opp_idx = _require_int(params, "opponent_idx")
        src = _require_coord(params, "opponent_card_coord")
        opp_board = state.players[opp_idx].board
        removed = opp_board.remove(src[0], src[1])
        state.discard.append(removed)
        return RewardApplyResult(True, False, False, None, "removed_opponent_card")

    if reward_id == "RWD7":
        opp_idx = _require_int(params, "opponent_idx")
        src = _require_coord(params, "opponent_card_coord")
        dst = _require_coord(params, "opponent_empty_coord")
        opp_board = state.players[opp_idx].board
        opp_board.move(src, dst)
        return RewardApplyResult(True, False, False, None, "moved_opponent_card")

    if reward_id == "RWD8":
        dst = _require_coord(params, "self_empty_coord")
        idx_from_top = _require_int(params, "discard_index_from_top")
        taken = _take_from_discard_by_index_from_top(state, idx_from_top)
        if taken is None:
            return RewardApplyResult(False, True, False, None, "discard_index_invalid_or_empty")
        if not me_board.is_empty(dst[0], dst[1]):
            raise RewardError("Destination not empty")
        me_board.place(dst[0], dst[1], taken)
        return RewardApplyResult(True, False, False, None, "took_any_discard")

    if reward_id == "RWD9":
        dst = _require_coord(params, "self_empty_coord")
        if not me_board.is_empty(dst[0], dst[1]):
            raise RewardError("Destination not empty")
        me_board.place(dst[0], dst[1], Joker())
        return RewardApplyResult(True, False, False, None, "spawned_joker")

    if reward_id == "RWD10":
        opp_idx = _require_int(params, "opponent_idx")
        src = _require_coord(params, "opponent_card_coord")
        dst = _require_coord(params, "self_empty_coord")
        if not me_board.is_empty(dst[0], dst[1]):
            raise RewardError("Destination not empty")
        opp_board = state.players[opp_idx].board
        stolen = opp_board.get(src[0], src[1])
        if stolen is None:
            raise RewardError("Source empty")
        opp_board.remove(src[0], src[1])
        me_board.place(dst[0], dst[1], stolen)
        return RewardApplyResult(True, False, False, None, "stole_opponent_card")

    raise RewardError(f"Unsupported reward_id: {reward_id}")


def _list_params_remove_opponent(
    state: GameState,
    cfg: RulesConfig,
    spec: RewardSpec,
    pid: int,
) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    forbid_source = cfg.game.rewards.opponent_actions_forbid_touch_protected_zone or spec.constraints.forbid_opponent_protected_zone_as_source

    for opp_idx in _opponent_indices(state, pid):
        opp_board = state.players[opp_idx].board
        for (x, y), _ in opp_board.occupied_cells():
            if forbid_source and state.is_opponent_cell_protected(opp_idx, x, y):
                continue
            params.append({"opponent_idx": opp_idx, "opponent_card_coord": (x, y)})
    return params


def _list_params_move_opponent(
    state: GameState,
    cfg: RulesConfig,
    spec: RewardSpec,
    pid: int,
) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []

    forbid_source = cfg.game.rewards.opponent_actions_forbid_touch_protected_zone or spec.constraints.forbid_opponent_protected_zone_as_source
    forbid_dest = cfg.game.rewards.opponent_actions_forbid_touch_protected_zone or spec.constraints.forbid_opponent_protected_zone_as_destination

    for opp_idx in _opponent_indices(state, pid):
        opp_board = state.players[opp_idx].board
        src_coords: List[Coord] = []
        dst_coords: List[Coord] = []

        for (x, y), _ in opp_board.occupied_cells():
            if forbid_source and state.is_opponent_cell_protected(opp_idx, x, y):
                continue
            src_coords.append((x, y))

        for (x, y) in opp_board.empty_cells():
            if forbid_dest and state.is_opponent_cell_protected(opp_idx, x, y):
                continue
            dst_coords.append((x, y))

        if not src_coords or not dst_coords:
            continue

        for src in src_coords:
            for dst in dst_coords:
                if src == dst:
                    continue
                if cfg.game.rewards.opponent_actions_forbid_move_into_or_out_of_protected_zone:
                    # Нельзя перемещать в/из защищенной зоны оппонента
                    src_prot = state.is_opponent_cell_protected(opp_idx, src[0], src[1])
                    dst_prot = state.is_opponent_cell_protected(opp_idx, dst[0], dst[1])
                    if src_prot != dst_prot:
                        continue
                params.append({"opponent_idx": opp_idx, "opponent_card_coord": src, "opponent_empty_coord": dst})

    return params


def _list_params_steal_opponent(
    state: GameState,
    cfg: RulesConfig,
    spec: RewardSpec,
    pid: int,
) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []

    me_board = state.players[pid].board
    empties = me_board.empty_cells()
    if not empties:
        return []

    forbid_source = cfg.game.rewards.opponent_actions_forbid_touch_protected_zone or spec.constraints.forbid_opponent_protected_zone_as_source

    for opp_idx in _opponent_indices(state, pid):
        opp_board = state.players[opp_idx].board
        for (x, y), _ in opp_board.occupied_cells():
            if forbid_source and state.is_opponent_cell_protected(opp_idx, x, y):
                continue
            for dst in empties:
                params.append({"opponent_idx": opp_idx, "opponent_card_coord": (x, y), "self_empty_coord": dst})

    return params


def _opponent_indices(state: GameState, pid: int) -> List[int]:
    return [i for i in range(len(state.players)) if i != pid]


def _take_from_discard_by_index_from_top(state: GameState, idx_from_top: int) -> Optional[PlacedCard]:
    if idx_from_top < 0:
        return None
    if len(state.discard) == 0:
        return None
    if idx_from_top >= len(state.discard):
        return None

    # 0 это верх, то есть discard[-1]
    real_index = len(state.discard) - 1 - idx_from_top
    return state.discard.pop(real_index)


def _require_coord(params: Dict[str, Any], key: str) -> Coord:
    if key not in params:
        raise RewardError(f"Missing param: {key}")
    v = params[key]
    if not isinstance(v, tuple) or len(v) != 2 or not all(isinstance(x, int) for x in v):
        raise RewardError(f"Param {key} must be Coord tuple[int,int], got {v}")
    return (int(v[0]), int(v[1]))


def _require_int(params: Dict[str, Any], key: str) -> int:
    if key not in params:
        raise RewardError(f"Missing param: {key}")
    v = params[key]
    if not isinstance(v, int):
        raise RewardError(f"Param {key} must be int, got {v}")
    return int(v)


def _params_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        if a[k] != b[k]:
            return False
    return True
