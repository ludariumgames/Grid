from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class ConfigError(ValueError):
    def __init__(self, message: str, path: str) -> None:
        super().__init__(f"{message} (at {path})")
        self.path = path


def _path(parent: str, key: str) -> str:
    if key.startswith("["):
        return f"{parent}{key}"
    return f"{parent}.{key}"


def _require_key(obj: Dict[str, Any], key: str, p: str) -> Any:
    if key not in obj:
        raise ConfigError(f"Missing required key '{key}'", _path(p, key))
    return obj[key]


def _expect_dict(value: Any, p: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError("Expected object/dict", p)
    return value


def _expect_list(value: Any, p: str) -> List[Any]:
    if not isinstance(value, list):
        raise ConfigError("Expected array/list", p)
    return value


def _expect_str(value: Any, p: str) -> str:
    if not isinstance(value, str):
        raise ConfigError("Expected string", p)
    return value


def _expect_int(value: Any, p: str) -> int:
    if not isinstance(value, int):
        raise ConfigError("Expected int", p)
    return value


def _expect_bool(value: Any, p: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError("Expected bool", p)
    return value


def _expect_optional_bool(value: Any, p: str) -> Optional[bool]:
    if value is None:
        return None
    return _expect_bool(value, p)


def _expect_in(value: str, allowed: Sequence[str], p: str) -> str:
    if value not in allowed:
        raise ConfigError(f"Invalid value '{value}', allowed: {list(allowed)}", p)
    return value


def _expect_unique_ids(items: Iterable[Any], id_getter, p: str) -> None:
    seen: set[str] = set()
    for idx, it in enumerate(items):
        item_id = id_getter(it)
        if item_id in seen:
            raise ConfigError(f"Duplicate id '{item_id}'", _path(p, f"[{idx}]"))
        seen.add(item_id)


@dataclass(frozen=True)
class EnumsSpec:
    pattern_shape: Tuple[str, ...]
    rank_rule: Tuple[str, ...]
    color_rule: Tuple[str, ...]
    reward_id: Tuple[str, ...]
    fallback_policy: Tuple[str, ...]


@dataclass(frozen=True)
class PlayersSpec:
    min_players: int
    max_players: int
    next_player_definition: str
    turn_direction: str


@dataclass(frozen=True)
class DeckSpec:
    ranks: Tuple[int, ...]
    colors: Tuple[int, ...]
    shapes: Tuple[int, ...]
    total_cards: int
    unique_cards: bool


@dataclass(frozen=True)
class ProtectedZoneSpec:
    position: str
    shape: str
    size: int


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    protected_zone: ProtectedZoneSpec


@dataclass(frozen=True)
class DraftSpec:
    reveal: int
    pick_self: int
    pass_next_player: int
    discard: int


@dataclass(frozen=True)
class GamePatternsSpec:
    auto_trigger: bool
    auto_clear_after_trigger: bool
    multi_trigger_resolution_order: str


@dataclass(frozen=True)
class GameRewardsSpec:
    optional_by_default: bool
    reward_placement_failure_does_not_end_game: bool
    opponent_actions_forbid_touch_protected_zone: bool
    opponent_actions_forbid_move_into_or_out_of_protected_zone: bool
    extra_turn_ignored_if_deck_empty: bool


@dataclass(frozen=True)
class JokerSpec:
    enabled: bool
    matches_any_rank: bool
    matches_any_color: bool


@dataclass(frozen=True)
class TerminationRules:
    end_if_deck_less_than_reveal_before_phase2: bool
    reveal_required_cards: int
    end_if_no_space_for_mandatory_hand_placement: bool
    end_if_no_space_for_mandatory_draft_placement: bool
    do_not_end_if_no_space_for_reward_placement: bool
    extra_turn_ignored_if_deck_empty: bool
    deck_empty_allows_phase1_for_next_players: bool


@dataclass(frozen=True)
class PatternRewardSpec:
    id: str
    optional: bool


@dataclass(frozen=True)
class PatternSpec:
    id: str
    shape: str
    size_cards: int
    rank_rule: str
    color_rule: str
    difficulty_rank: int
    vp: int
    reward: Optional[PatternRewardSpec]
    source_ref: Dict[str, Any]


@dataclass(frozen=True)
class RewardConstraintsSpec:
    requires_free_cell_on_self_board: bool
    forbid_opponent_protected_zone_as_source: bool
    forbid_opponent_protected_zone_as_destination: bool


@dataclass(frozen=True)
class RewardSpec:
    id: str
    action_type: str
    target_scope: str
    optional: bool
    constraints: RewardConstraintsSpec
    agent_choices: Tuple[str, ...]


@dataclass(frozen=True)
class AgentContractSpec:
    fallback_policy_on_invalid_choice: str


@dataclass(frozen=True)
class GameSpec:
    players: PlayersSpec
    deck: DeckSpec
    grid: GridSpec
    draft: DraftSpec
    patterns: GamePatternsSpec
    rewards: GameRewardsSpec
    joker: JokerSpec


@dataclass(frozen=True)
class RulesConfig:
    version: str
    game: GameSpec
    enums: EnumsSpec
    patterns: Tuple[PatternSpec, ...]
    rewards: Tuple[RewardSpec, ...]
    termination: TerminationRules
    agent_contract: AgentContractSpec


def load_rules_config(path: str | Path) -> RulesConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return _parse_rules_config(raw)


def _parse_rules_config(raw: Any) -> RulesConfig:
    root = "$"
    obj = _expect_dict(raw, root)

    version = _expect_str(_require_key(obj, "version", root), _path(root, "version"))

    enums = _parse_enums(_require_key(obj, "enums", root), _path(root, "enums"))
    game = _parse_game(_require_key(obj, "game", root), _path(root, "game"))
    termination = _parse_termination(
        _require_key(obj, "termination", root),
        _path(root, "termination"),
    )
    agent_contract = _parse_agent_contract(
        _require_key(obj, "agent_contract", root),
        _path(root, "agent_contract"),
        enums,
    )

    patterns = _parse_patterns(
        _require_key(obj, "patterns", root),
        _path(root, "patterns"),
        enums,
    )
    rewards = _parse_rewards(
        _require_key(obj, "rewards", root),
        _path(root, "rewards"),
        enums,
    )

    _validate_cross_refs(enums, patterns, rewards, game, termination)

    return RulesConfig(
        version=version,
        game=game,
        enums=enums,
        patterns=tuple(patterns),
        rewards=tuple(rewards),
        termination=termination,
        agent_contract=agent_contract,
    )


def _parse_enums(raw: Any, p: str) -> EnumsSpec:
    obj = _expect_dict(raw, p)

    pattern_shape = tuple(_expect_list(_require_key(obj, "pattern_shape", p), _path(p, "pattern_shape")))
    rank_rule = tuple(_expect_list(_require_key(obj, "rank_rule", p), _path(p, "rank_rule")))
    color_rule = tuple(_expect_list(_require_key(obj, "color_rule", p), _path(p, "color_rule")))
    reward_id = tuple(_expect_list(_require_key(obj, "reward_id", p), _path(p, "reward_id")))
    fallback_policy = tuple(_expect_list(_require_key(obj, "fallback_policy", p), _path(p, "fallback_policy")))

    for i, v in enumerate(pattern_shape):
        _expect_str(v, _path(p, f"pattern_shape[{i}]"))
    for i, v in enumerate(rank_rule):
        _expect_str(v, _path(p, f"rank_rule[{i}]"))
    for i, v in enumerate(color_rule):
        _expect_str(v, _path(p, f"color_rule[{i}]"))
    for i, v in enumerate(reward_id):
        _expect_str(v, _path(p, f"reward_id[{i}]"))
    for i, v in enumerate(fallback_policy):
        _expect_str(v, _path(p, f"fallback_policy[{i}]"))

    return EnumsSpec(
        pattern_shape=tuple(pattern_shape),
        rank_rule=tuple(rank_rule),
        color_rule=tuple(color_rule),
        reward_id=tuple(reward_id),
        fallback_policy=tuple(fallback_policy),
    )


def _parse_game(raw: Any, p: str) -> GameSpec:
    obj = _expect_dict(raw, p)

    players = _parse_game_players(_require_key(obj, "players", p), _path(p, "players"))
    deck = _parse_game_deck(_require_key(obj, "deck", p), _path(p, "deck"))
    grid = _parse_game_grid(_require_key(obj, "grid", p), _path(p, "grid"))
    draft = _parse_game_draft(_require_key(obj, "draft", p), _path(p, "draft"))
    patterns = _parse_game_patterns(_require_key(obj, "patterns", p), _path(p, "patterns"))
    rewards = _parse_game_rewards(_require_key(obj, "rewards", p), _path(p, "rewards"))
    joker = _parse_game_joker(_require_key(obj, "joker", p), _path(p, "joker"))

    return GameSpec(
        players=players,
        deck=deck,
        grid=grid,
        draft=draft,
        patterns=patterns,
        rewards=rewards,
        joker=joker,
    )


def _parse_game_players(raw: Any, p: str) -> PlayersSpec:
    obj = _expect_dict(raw, p)
    min_players = _expect_int(_require_key(obj, "min_players", p), _path(p, "min_players"))
    max_players = _expect_int(_require_key(obj, "max_players", p), _path(p, "max_players"))
    next_player_definition = _expect_str(
        _require_key(obj, "next_player_definition", p),
        _path(p, "next_player_definition"),
    )
    turn_direction = _expect_str(_require_key(obj, "turn_direction", p), _path(p, "turn_direction"))

    if min_players < 2 or max_players > 4 or min_players > max_players:
        raise ConfigError("Invalid players range", p)

    return PlayersSpec(
        min_players=min_players,
        max_players=max_players,
        next_player_definition=next_player_definition,
        turn_direction=turn_direction,
    )


def _parse_game_deck(raw: Any, p: str) -> DeckSpec:
    obj = _expect_dict(raw, p)

    ranks = _expect_list(_require_key(obj, "ranks", p), _path(p, "ranks"))
    colors = _expect_list(_require_key(obj, "colors", p), _path(p, "colors"))
    shapes = _expect_list(_require_key(obj, "shapes", p), _path(p, "shapes"))

    ranks_i = tuple(_expect_int(v, _path(p, f"ranks[{i}]")) for i, v in enumerate(ranks))
    colors_i = tuple(_expect_int(v, _path(p, f"colors[{i}]")) for i, v in enumerate(colors))
    shapes_i = tuple(_expect_int(v, _path(p, f"shapes[{i}]")) for i, v in enumerate(shapes))

    total_cards = _expect_int(_require_key(obj, "total_cards", p), _path(p, "total_cards"))
    unique_cards = _expect_bool(_require_key(obj, "unique_cards", p), _path(p, "unique_cards"))

    if total_cards <= 0:
        raise ConfigError("total_cards must be > 0", _path(p, "total_cards"))

    return DeckSpec(
        ranks=ranks_i,
        colors=colors_i,
        shapes=shapes_i,
        total_cards=total_cards,
        unique_cards=unique_cards,
    )


def _parse_game_grid(raw: Any, p: str) -> GridSpec:
    obj = _expect_dict(raw, p)
    width = _expect_int(_require_key(obj, "width", p), _path(p, "width"))
    height = _expect_int(_require_key(obj, "height", p), _path(p, "height"))

    pz_raw = _require_key(obj, "protected_zone", p)
    pz_obj = _expect_dict(pz_raw, _path(p, "protected_zone"))
    pz_pos = _expect_str(_require_key(pz_obj, "position", _path(p, "protected_zone")), _path(p, "protected_zone.position"))
    pz_shape = _expect_str(_require_key(pz_obj, "shape", _path(p, "protected_zone")), _path(p, "protected_zone.shape"))
    pz_size = _expect_int(_require_key(pz_obj, "size", _path(p, "protected_zone")), _path(p, "protected_zone.size"))

    if width <= 0 or height <= 0:
        raise ConfigError("Grid width/height must be > 0", p)

    protected_zone = ProtectedZoneSpec(position=pz_pos, shape=pz_shape, size=pz_size)
    return GridSpec(width=width, height=height, protected_zone=protected_zone)


def _parse_game_draft(raw: Any, p: str) -> DraftSpec:
    obj = _expect_dict(raw, p)
    reveal = _expect_int(_require_key(obj, "reveal", p), _path(p, "reveal"))
    pick_self = _expect_int(_require_key(obj, "pick_self", p), _path(p, "pick_self"))
    pass_next_player = _expect_int(_require_key(obj, "pass_next_player", p), _path(p, "pass_next_player"))
    discard = _expect_int(_require_key(obj, "discard", p), _path(p, "discard"))

    if reveal <= 0 or pick_self < 0 or pass_next_player < 0 or discard < 0:
        raise ConfigError("Invalid draft counts", p)
    if pick_self + pass_next_player + discard != reveal:
        raise ConfigError("Draft counts must sum to reveal", p)

    return DraftSpec(reveal=reveal, pick_self=pick_self, pass_next_player=pass_next_player, discard=discard)


def _parse_game_patterns(raw: Any, p: str) -> GamePatternsSpec:
    obj = _expect_dict(raw, p)
    auto_trigger = _expect_bool(_require_key(obj, "auto_trigger", p), _path(p, "auto_trigger"))
    auto_clear = _expect_bool(_require_key(obj, "auto_clear_after_trigger", p), _path(p, "auto_clear_after_trigger"))
    order = _expect_str(_require_key(obj, "multi_trigger_resolution_order", p), _path(p, "multi_trigger_resolution_order"))
    return GamePatternsSpec(
        auto_trigger=auto_trigger,
        auto_clear_after_trigger=auto_clear,
        multi_trigger_resolution_order=order,
    )


def _parse_game_rewards(raw: Any, p: str) -> GameRewardsSpec:
    obj = _expect_dict(raw, p)
    optional_by_default = _expect_bool(_require_key(obj, "optional_by_default", p), _path(p, "optional_by_default"))
    placement_fail_no_end = _expect_bool(
        _require_key(obj, "reward_placement_failure_does_not_end_game", p),
        _path(p, "reward_placement_failure_does_not_end_game"),
    )
    forbid_touch = _expect_bool(
        _require_key(obj, "opponent_actions_forbid_touch_protected_zone", p),
        _path(p, "opponent_actions_forbid_touch_protected_zone"),
    )
    forbid_move_inout = _expect_bool(
        _require_key(obj, "opponent_actions_forbid_move_into_or_out_of_protected_zone", p),
        _path(p, "opponent_actions_forbid_move_into_or_out_of_protected_zone"),
    )
    extra_turn_ignored = _expect_bool(
        _require_key(obj, "extra_turn_ignored_if_deck_empty", p),
        _path(p, "extra_turn_ignored_if_deck_empty"),
    )
    return GameRewardsSpec(
        optional_by_default=optional_by_default,
        reward_placement_failure_does_not_end_game=placement_fail_no_end,
        opponent_actions_forbid_touch_protected_zone=forbid_touch,
        opponent_actions_forbid_move_into_or_out_of_protected_zone=forbid_move_inout,
        extra_turn_ignored_if_deck_empty=extra_turn_ignored,
    )


def _parse_game_joker(raw: Any, p: str) -> JokerSpec:
    obj = _expect_dict(raw, p)
    enabled = _expect_bool(_require_key(obj, "enabled", p), _path(p, "enabled"))
    matches_any_rank = _expect_bool(_require_key(obj, "matches_any_rank", p), _path(p, "matches_any_rank"))
    matches_any_color = _expect_bool(_require_key(obj, "matches_any_color", p), _path(p, "matches_any_color"))
    return JokerSpec(enabled=enabled, matches_any_rank=matches_any_rank, matches_any_color=matches_any_color)


def _parse_termination(raw: Any, p: str) -> TerminationRules:
    obj = _expect_dict(raw, p)

    end_if_deck_lt = _expect_bool(
        _require_key(obj, "end_if_deck_less_than_reveal_before_phase2", p),
        _path(p, "end_if_deck_less_than_reveal_before_phase2"),
    )
    reveal_required = _expect_int(_require_key(obj, "reveal_required_cards", p), _path(p, "reveal_required_cards"))
    end_no_space_hand = _expect_bool(
        _require_key(obj, "end_if_no_space_for_mandatory_hand_placement", p),
        _path(p, "end_if_no_space_for_mandatory_hand_placement"),
    )
    end_no_space_draft = _expect_bool(
        _require_key(obj, "end_if_no_space_for_mandatory_draft_placement", p),
        _path(p, "end_if_no_space_for_mandatory_draft_placement"),
    )
    do_not_end_reward = _expect_bool(
        _require_key(obj, "do_not_end_if_no_space_for_reward_placement", p),
        _path(p, "do_not_end_if_no_space_for_reward_placement"),
    )
    extra_turn_ignored = _expect_bool(
        _require_key(obj, "extra_turn_ignored_if_deck_empty", p),
        _path(p, "extra_turn_ignored_if_deck_empty"),
    )
    deck_empty_allows_phase1 = _expect_bool(
        _require_key(obj, "deck_empty_allows_phase1_for_next_players", p),
        _path(p, "deck_empty_allows_phase1_for_next_players"),
    )

    if reveal_required <= 0:
        raise ConfigError("reveal_required_cards must be > 0", _path(p, "reveal_required_cards"))

    return TerminationRules(
        end_if_deck_less_than_reveal_before_phase2=end_if_deck_lt,
        reveal_required_cards=reveal_required,
        end_if_no_space_for_mandatory_hand_placement=end_no_space_hand,
        end_if_no_space_for_mandatory_draft_placement=end_no_space_draft,
        do_not_end_if_no_space_for_reward_placement=do_not_end_reward,
        extra_turn_ignored_if_deck_empty=extra_turn_ignored,
        deck_empty_allows_phase1_for_next_players=deck_empty_allows_phase1,
    )


def _parse_agent_contract(raw: Any, p: str, enums: EnumsSpec) -> AgentContractSpec:
    obj = _expect_dict(raw, p)
    fp = _expect_str(
        _require_key(obj, "fallback_policy_on_invalid_choice", p),
        _path(p, "fallback_policy_on_invalid_choice"),
    )
    _expect_in(fp, enums.fallback_policy, _path(p, "fallback_policy_on_invalid_choice"))
    return AgentContractSpec(fallback_policy_on_invalid_choice=fp)


def _parse_patterns(raw: Any, p: str, enums: EnumsSpec) -> List[PatternSpec]:
    arr = _expect_list(raw, p)
    patterns: List[PatternSpec] = []

    for i, item in enumerate(arr):
        ip = _path(p, f"[{i}]")
        obj = _expect_dict(item, ip)

        pid = _expect_str(_require_key(obj, "id", ip), _path(ip, "id"))
        shape = _expect_in(
            _expect_str(_require_key(obj, "shape", ip), _path(ip, "shape")),
            enums.pattern_shape,
            _path(ip, "shape"),
        )
        size_cards = _expect_int(_require_key(obj, "size_cards", ip), _path(ip, "size_cards"))
        rank_rule = _expect_in(
            _expect_str(_require_key(obj, "rank_rule", ip), _path(ip, "rank_rule")),
            enums.rank_rule,
            _path(ip, "rank_rule"),
        )
        color_rule = _expect_in(
            _expect_str(_require_key(obj, "color_rule", ip), _path(ip, "color_rule")),
            enums.color_rule,
            _path(ip, "color_rule"),
        )
        difficulty_rank = _expect_int(_require_key(obj, "difficulty_rank", ip), _path(ip, "difficulty_rank"))
        vp = _expect_int(_require_key(obj, "vp", ip), _path(ip, "vp"))

        reward_val = obj.get("reward")
        reward: Optional[PatternRewardSpec] = None
        if reward_val is not None:
            rp = _path(ip, "reward")
            robj = _expect_dict(reward_val, rp)
            rid = _expect_in(
                _expect_str(_require_key(robj, "id", rp), _path(rp, "id")),
                enums.reward_id,
                _path(rp, "id"),
            )
            optional = _expect_bool(_require_key(robj, "optional", rp), _path(rp, "optional"))
            reward = PatternRewardSpec(id=rid, optional=optional)

        source_ref_val = obj.get("source_ref", {})
        source_ref = _expect_dict(source_ref_val, _path(ip, "source_ref"))

        if size_cards <= 0:
            raise ConfigError("size_cards must be > 0", _path(ip, "size_cards"))

        patterns.append(
            PatternSpec(
                id=pid,
                shape=shape,
                size_cards=size_cards,
                rank_rule=rank_rule,
                color_rule=color_rule,
                difficulty_rank=difficulty_rank,
                vp=vp,
                reward=reward,
                source_ref=source_ref,
            )
        )

    _expect_unique_ids(patterns, lambda x: x.id, p)
    return patterns


def _parse_rewards(raw: Any, p: str, enums: EnumsSpec) -> List[RewardSpec]:
    arr = _expect_list(raw, p)
    rewards: List[RewardSpec] = []

    for i, item in enumerate(arr):
        ip = _path(p, f"[{i}]")
        obj = _expect_dict(item, ip)

        rid = _expect_in(
            _expect_str(_require_key(obj, "id", ip), _path(ip, "id")),
            enums.reward_id,
            _path(ip, "id"),
        )
        action_type = _expect_str(_require_key(obj, "action_type", ip), _path(ip, "action_type"))
        target_scope = _expect_str(_require_key(obj, "target_scope", ip), _path(ip, "target_scope"))
        optional = _expect_bool(_require_key(obj, "optional", ip), _path(ip, "optional"))

        c_raw = _require_key(obj, "constraints", ip)
        c_obj = _expect_dict(c_raw, _path(ip, "constraints"))
        constraints = RewardConstraintsSpec(
            requires_free_cell_on_self_board=_expect_bool(
                _require_key(c_obj, "requires_free_cell_on_self_board", _path(ip, "constraints")),
                _path(ip, "constraints.requires_free_cell_on_self_board"),
            ),
            forbid_opponent_protected_zone_as_source=_expect_bool(
                _require_key(c_obj, "forbid_opponent_protected_zone_as_source", _path(ip, "constraints")),
                _path(ip, "constraints.forbid_opponent_protected_zone_as_source"),
            ),
            forbid_opponent_protected_zone_as_destination=_expect_bool(
                _require_key(c_obj, "forbid_opponent_protected_zone_as_destination", _path(ip, "constraints")),
                _path(ip, "constraints.forbid_opponent_protected_zone_as_destination"),
            ),
        )

        agent_choices_raw = _require_key(obj, "agent_choices", ip)
        agent_choices_list = _expect_list(agent_choices_raw, _path(ip, "agent_choices"))
        agent_choices = tuple(_expect_str(v, _path(ip, f"agent_choices[{j}]")) for j, v in enumerate(agent_choices_list))

        rewards.append(
            RewardSpec(
                id=rid,
                action_type=action_type,
                target_scope=target_scope,
                optional=optional,
                constraints=constraints,
                agent_choices=agent_choices,
            )
        )

    _expect_unique_ids(rewards, lambda x: x.id, p)
    return rewards


def _validate_cross_refs(
    enums: EnumsSpec,
    patterns: Sequence[PatternSpec],
    rewards: Sequence[RewardSpec],
    game: GameSpec,
    termination: TerminationRules,
) -> None:
    reward_ids_defined = {r.id for r in rewards}

    for i, pat in enumerate(patterns):
        if pat.reward is not None:
            if pat.reward.id not in reward_ids_defined:
                raise ConfigError(
                    f"Pattern refers to missing reward id '{pat.reward.id}'",
                    f"$.patterns[{i}].reward.id",
                )

    if termination.reveal_required_cards != game.draft.reveal:
        raise ConfigError(
            "termination.reveal_required_cards must match game.draft.reveal",
            "$.termination.reveal_required_cards",
        )

    if game.deck.unique_cards and game.deck.total_cards != len(game.deck.ranks) * len(game.deck.colors) * len(game.deck.shapes):
        raise ConfigError(
            "Deck total_cards does not match ranks*colors*shapes for unique_cards=true",
            "$.game.deck.total_cards",
        )
