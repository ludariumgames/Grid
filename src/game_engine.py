from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.base import Agent, RewardContext
from config_loader import RulesConfig
from models import (
    Card,
    Coord,
    FallbackEvent,
    GameState,
    MatchAndRewardResult,
    PlacedCard,
)
from pattern_engine import find_patterns_on_board
from reward_engine import RewardApplyResult, apply_reward, list_valid_reward_params


END1 = "END1"
END2 = "END2"


@dataclass(slots=True)
class GameRunStats:
    pattern_triggers: Dict[str, int] = field(default_factory=dict)
    reward_applied: Dict[str, int] = field(default_factory=dict)
    reward_refused: Dict[str, int] = field(default_factory=dict)
    reward_impossible: Dict[str, int] = field(default_factory=dict)
    fallback_events: List[FallbackEvent] = field(default_factory=list)
    turns_taken: int = 0
    end_reason: Optional[str] = None


class GameEngine:
    def __init__(
        self,
        cfg: RulesConfig,
        agents: Sequence[Agent],
        rng: random.Random,
    ) -> None:
        self.cfg = cfg
        self.agents = list(agents)
        self.rng = rng

        fp = cfg.agent_contract.fallback_policy_on_invalid_choice
        if fp != "random_valid_with_logging":
            raise ValueError(f"Unsupported fallback policy: {fp}")

        if cfg.game.patterns.multi_trigger_resolution_order != "agent_choice":
            raise ValueError(
                f"Unsupported multi_trigger_resolution_order: {cfg.game.patterns.multi_trigger_resolution_order}"
            )

    def play_game(self, state: GameState) -> Tuple[GameState, GameRunStats, List[MatchAndRewardResult]]:
        if len(self.agents) != len(state.players):
            raise ValueError("Number of agents must match number of players")

        stats = GameRunStats()
        events: List[MatchAndRewardResult] = []

        # True значит: следующий ход снова делает тот же игрок (один раз).
        extra_turn_pending: bool = False

        while not state.ended:
            pid = state.current_player_idx
            agent = self.agents[pid]

            stats.turns_taken += 1
            state.turn_number += 1

            # skip_turn: пропускаем весь ход целиком (фаза 1 и фаза 2)
            if state.players[pid].skip_next_turn:
                state.players[pid].skip_next_turn = False
                state.current_player_idx = state.next_player_idx()
                extra_turn_pending = False
                continue

            extra_turn_awarded_this_turn = False

            ok1, extra1 = self._phase1_place_hand(state, agent, stats, events)
            if not ok1:
                break
            extra_turn_awarded_this_turn = extra_turn_awarded_this_turn or extra1

            ok2, extra2 = self._phase2_draft(state, agent, stats, events)
            if not ok2:
                break
            extra_turn_awarded_this_turn = extra_turn_awarded_this_turn or extra2

            # Если этот ход дал extra_turn, следующий ход делает тот же игрок.
            # При этом extra_turn не копится внутри одного хода.
            extra_turn_pending = extra_turn_awarded_this_turn

            if extra_turn_pending:
                extra_turn_pending = False
                # не меняем игрока
            else:
                state.current_player_idx = state.next_player_idx()

        stats.end_reason = state.end_reason
        return state, stats, events

    def _phase1_place_hand(
        self,
        state: GameState,
        agent: Agent,
        stats: GameRunStats,
        events: List[MatchAndRewardResult],
    ) -> Tuple[bool, bool]:
        pid = state.current_player_idx
        me = state.players[pid]

        extra_turn = False

        while me.hand:
            card = me.hand.pop(0)

            if self.cfg.termination.end_if_no_space_for_mandatory_hand_placement and not me.board.has_space():
                self._end_game(state, END2)
                return False, extra_turn

            cell = agent.choose_placement_cell(state, card)
            valid_cells = me.board.empty_cells()
            if not self._is_valid_coord(cell, valid_cells):
                chosen = self.rng.choice(valid_cells) if valid_cells else (0, 0)
                stats.fallback_events.append(
                    self._fallback_event(
                        state,
                        agent,
                        decision_type="placement_cell",
                        reason=f"invalid_cell_returned:{cell}",
                        chosen=str(chosen),
                    )
                )
                cell = chosen

            me.board.place(cell[0], cell[1], card)

            extra_turn = extra_turn or self._resolve_autopatterns_loop(state, agent, stats, events)

        return True, extra_turn

    def _phase2_draft(
        self,
        state: GameState,
        agent: Agent,
        stats: GameRunStats,
        events: List[MatchAndRewardResult],
    ) -> Tuple[bool, bool]:
        pid = state.current_player_idx
        me = state.players[pid]

        if self.cfg.termination.end_if_deck_less_than_reveal_before_phase2:
            if len(state.deck) < self.cfg.termination.reveal_required_cards:
                self._end_game(state, END1)
                return False, False

        revealed = self._draw_n(state, self.cfg.game.draft.reveal)
        if len(revealed) != self.cfg.game.draft.reveal:
            self._end_game(state, END1)
            return False, False

        pick_idx = agent.choose_draft_pick(state, revealed)
        if not self._is_valid_index(pick_idx, len(revealed)):
            chosen = self.rng.randrange(len(revealed))
            stats.fallback_events.append(
                self._fallback_event(
                    state,
                    agent,
                    decision_type="draft_pick",
                    reason=f"invalid_index_returned:{pick_idx}",
                    chosen=str(chosen),
                )
            )
            pick_idx = chosen

        picked = revealed.pop(pick_idx)

        pass_idx = agent.choose_draft_pass(state, revealed)
        if not self._is_valid_index(pass_idx, len(revealed)):
            chosen = self.rng.randrange(len(revealed))
            stats.fallback_events.append(
                self._fallback_event(
                    state,
                    agent,
                    decision_type="draft_pass",
                    reason=f"invalid_index_returned:{pass_idx}",
                    chosen=str(chosen),
                )
            )
            pass_idx = chosen

        pass_card = revealed.pop(pass_idx)
        discard_card = revealed[0]

        left_idx = state.player_left_of(pid)
        state.players[left_idx].hand.append(pass_card)

        state.discard.append(discard_card)

        if self.cfg.termination.end_if_no_space_for_mandatory_draft_placement and not me.board.has_space():
            self._end_game(state, END2)
            return False, False

        cell = agent.choose_placement_cell(state, picked)
        valid_cells = me.board.empty_cells()
        if not self._is_valid_coord(cell, valid_cells):
            chosen = self.rng.choice(valid_cells) if valid_cells else (0, 0)
            stats.fallback_events.append(
                self._fallback_event(
                    state,
                    agent,
                    decision_type="placement_cell",
                    reason=f"invalid_cell_returned:{cell}",
                    chosen=str(chosen),
                )
            )
            cell = chosen

        me.board.place(cell[0], cell[1], picked)

        extra_turn = self._resolve_autopatterns_loop(state, agent, stats, events)
        return True, extra_turn

    def _resolve_autopatterns_loop(
        self,
        state: GameState,
        agent: Agent,
        stats: GameRunStats,
        events: List[MatchAndRewardResult],
    ) -> bool:
        pid = state.current_player_idx
        me = state.players[pid]

        extra_turn_granted = False

        while True:
            found = find_patterns_on_board(me.board, self.cfg)
            if not found:
                return extra_turn_granted

            choice_idx = agent.choose_pattern_to_resolve(state, found)
            if not self._is_valid_index(choice_idx, len(found)):
                chosen = self.rng.randrange(len(found))
                stats.fallback_events.append(
                    self._fallback_event(
                        state,
                        agent,
                        decision_type="pattern_to_resolve",
                        reason=f"invalid_index_returned:{choice_idx}",
                        chosen=str(chosen),
                    )
                )
                choice_idx = chosen

            match = found[choice_idx]
            self._inc(stats.pattern_triggers, match.pattern_id)

            gained_vp = match.vp
            me.vp += gained_vp

            reward_id = match.reward_id
            reward_applied = False
            reward_refused = False
            reward_impossible = False

            pattern_objects: List[PlacedCard] = []
            for (x, y) in match.cells:
                c = me.board.get(x, y)
                if c is not None:
                    pattern_objects.append(c)

            if reward_id is not None:
                context = RewardContext(
                    pattern_id=match.pattern_id,
                    triggering_cells=match.cells,
                    deck_size=len(state.deck),
                    discard_size=len(state.discard),
                )

                apply_decision = agent.choose_apply_reward(state, reward_id, context)
                if not isinstance(apply_decision, bool):
                    chosen_bool = bool(self.rng.getrandbits(1))
                    stats.fallback_events.append(
                        self._fallback_event(
                            state,
                            agent,
                            decision_type="apply_reward",
                            reason=f"non_bool_returned:{apply_decision}",
                            chosen=str(chosen_bool),
                        )
                    )
                    apply_decision = chosen_bool

                if not apply_decision:
                    reward_refused = True
                    self._inc(stats.reward_refused, reward_id)
                else:
                    valid_params = list_valid_reward_params(state, self.cfg, reward_id, pid)
                    if not valid_params:
                        reward_impossible = True
                        self._inc(stats.reward_impossible, reward_id)
                    else:
                        params = agent.choose_reward_params(state, reward_id, context)
                        if not isinstance(params, dict):
                            params = {}

                        if not any(self._params_equal(vp, params) for vp in valid_params):
                            chosen_params = self.rng.choice(valid_params)
                            stats.fallback_events.append(
                                self._fallback_event(
                                    state,
                                    agent,
                                    decision_type="reward_params",
                                    reason=f"invalid_params_returned:{params}",
                                    chosen=str(chosen_params),
                                )
                            )
                            params = chosen_params

                        try:
                            res = apply_reward(state, self.cfg, reward_id, params, pid)
                        except Exception as e:
                            reward_impossible = True
                            self._inc(stats.reward_impossible, reward_id)
                            res = RewardApplyResult(
                                applied=False,
                                impossible=True,
                                extra_turn_granted=False,
                                skip_turn_target_idx=None,
                                notes=f"exception:{type(e).__name__}",
                            )

                        if res.impossible:
                            reward_impossible = True
                            self._inc(stats.reward_impossible, reward_id)
                        elif res.applied:
                            reward_applied = True
                            self._inc(stats.reward_applied, reward_id)

                        if res.extra_turn_granted:
                            extra_turn_granted = True

            for obj in pattern_objects:
                coord = self._find_card_object(me.board, obj)
                if coord is None:
                    continue
                removed = me.board.remove(coord[0], coord[1])
                state.discard.append(removed)

            events.append(
                MatchAndRewardResult(
                    player_idx=pid,
                    pattern_id=match.pattern_id,
                    gained_vp=gained_vp,
                    reward_id=reward_id,
                    reward_applied=reward_applied,
                    reward_refused=reward_refused,
                    reward_impossible=reward_impossible,
                )
            )

    def _draw_n(self, state: GameState, n: int) -> List[Card]:
        out: List[Card] = []
        for _ in range(n):
            if not state.deck:
                break
            out.append(state.deck.pop())
        return out

    def _end_game(self, state: GameState, reason: str) -> None:
        state.ended = True
        state.end_reason = reason

    def _fallback_event(
        self,
        state: GameState,
        agent: Agent,
        decision_type: str,
        reason: str,
        chosen: str,
    ) -> FallbackEvent:
        return FallbackEvent(
            game_id=state.game_id,
            turn_number=state.turn_number,
            player_idx=state.current_player_idx,
            agent_name=agent.name,
            decision_type=decision_type,
            reason=reason,
            chosen=chosen,
        )

    def _inc(self, d: Dict[str, int], key: str) -> None:
        d[key] = d.get(key, 0) + 1

    def _is_valid_index(self, idx: Any, length: int) -> bool:
        return isinstance(idx, int) and 0 <= idx < length

    def _is_valid_coord(self, coord: Any, valid: Sequence[Coord]) -> bool:
        return isinstance(coord, tuple) and len(coord) == 2 and coord in valid

    def _find_card_object(self, board, obj: PlacedCard) -> Optional[Coord]:
        for (x, y), c in board.occupied_cells():
            if c is obj:
                return (x, y)
        return None

    def _params_equal(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if a.keys() != b.keys():
            return False
        for k in a.keys():
            if a[k] != b[k]:
                return False
        return True
