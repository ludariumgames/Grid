import random
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Sequence

# Allow tests to import modules from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.base import Agent, RewardContext
from config_loader import load_rules_config
from game_engine import GameEngine
from game_setup import create_initial_game_state
from models import Card, Coord, GameState, PatternMatch


class ScriptedAgent(Agent):
    def __init__(self, name: str, placement_map: Dict[str, Coord]) -> None:
        self._name = name
        self._placement_map = placement_map

    @property
    def name(self) -> str:
        return self._name

    def choose_placement_cell(self, state: GameState, card: Card) -> Coord:
        if card.uid in self._placement_map:
            return self._placement_map[card.uid]
        empties = state.current_player().board.empty_cells()
        return empties[0] if empties else (0, 0)

    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        for i, c in enumerate(revealed_cards):
            if c.uid == "need_r4c1":
                return i
        return 0

    def choose_draft_pass(self, state: GameState, remaining_cards: Sequence[Card]) -> int:
        return 0

    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[PatternMatch]) -> int:
        for i, m in enumerate(found_patterns):
            if m.pattern_id == "H":
                return i
        return 0

    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        return reward_id == "RWD4"

    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        return {}


class TestExtraTurn(unittest.TestCase):
    def test_rwd4_grants_immediate_extra_turn(self) -> None:
        cfg = load_rules_config("config/game_rules_config_v0_1.json")

        # Create a deterministic state, then overwrite the deck with a controlled one.
        state = create_initial_game_state(cfg, num_players=2, rng=random.Random(999), game_id=1)

        # Pre-build almost-complete pattern H on player 0: 1-2-3-(need) with same color.
        b0 = state.players[0].board
        b0.place(0, 0, Card(rank=1, color=1, shape=1, uid="h_r1c1"))
        b0.place(1, 0, Card(rank=2, color=1, shape=1, uid="h_r2c1"))
        b0.place(2, 0, Card(rank=3, color=1, shape=1, uid="h_r3c1"))

        # Controlled deck:
        # Top of deck is the end of the list (state.deck.pop()).
        # We want the first draft reveal (3 cards) to include "need_r4c1",
        # and after drawing 3 cards, deck must still have 2 cards (so deck is not empty),
        # but < 3, so the next attempt to start phase 2 will end the game (END1).
        state.deck = [
            Card(rank=6, color=5, shape=3, uid="bottom_1"),
            Card(rank=6, color=4, shape=3, uid="bottom_2"),
            Card(rank=1, color=2, shape=1, uid="rev_1"),
            Card(rank=4, color=1, shape=1, uid="need_r4c1"),
            Card(rank=6, color=3, shape=1, uid="rev_3"),
        ]

        agents = [
            ScriptedAgent("scripted_p0", placement_map={"need_r4c1": (3, 0)}),
            ScriptedAgent("scripted_p1", placement_map={}),
        ]

        engine = GameEngine(cfg, agents, rng=random.Random(123))
        final_state, stats, _events = engine.play_game(state)

        # Expected flow:
        # - Turn 1: player 0 completes pattern H and applies RWD4.
        # - Turn 2: player 0 again (extra turn), but phase 2 cannot start because deck has only 2 cards -> END1.
        self.assertTrue(final_state.ended)
        self.assertEqual(final_state.end_reason, "END1")
        self.assertEqual(stats.turns_taken, 2)
        self.assertEqual(final_state.current_player_idx, 0)

        self.assertEqual(final_state.players[0].vp, 7)
        self.assertEqual(stats.pattern_triggers.get("H", 0), 1)
        self.assertEqual(stats.reward_applied.get("RWD4", 0), 1)


if __name__ == "__main__":
    unittest.main()
