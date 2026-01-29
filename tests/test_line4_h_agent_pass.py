import random
import unittest
from pathlib import Path

from config_loader import load_rules_config
from game_setup import create_initial_game_state
from models import Card

from agents.line4_h_agent import Line4HAgent


def _cfg_path() -> str:
    # tests/.. -> project root
    root = Path(__file__).resolve().parents[1]
    p = root / "config" / "game_rules_config_v0_1.json"
    if not p.exists():
        raise FileNotFoundError(f"Config not found at: {p}")
    return str(p)


class TestLine4HAgentDraftPass(unittest.TestCase):
    def test_pass_avoids_completing_opponent_I(self) -> None:
        cfg = load_rules_config(_cfg_path())

        # детерминированное состояние
        state = create_initial_game_state(cfg, num_players=2, rng=random.Random(1), game_id=1)
        state.current_player_idx = 0

        # Подготовим доску соперника (player 1): 1-2-3 в ряд, разные цвета, ждём 4-ку
        b = state.players[1].board
        b.place(0, 0, Card(rank=1, color=1, shape=1, uid="t1"))  # 1A
        b.place(1, 0, Card(rank=2, color=2, shape=1, uid="t2"))  # 2B
        b.place(2, 0, Card(rank=3, color=3, shape=1, uid="t3"))  # 3C
        # клетка (3,0) пустая, туда 4D закроет pattern I из конфига (rank seq, all colors distinct)

        remaining = [
            Card(rank=4, color=4, shape=1, uid="good_4D"),  # очень полезно сопернику
            Card(rank=6, color=1, shape=1, uid="bad_6A"),   # условный мусор
        ]

        agent = Line4HAgent(random.Random(0), cfg)

        pass_idx = agent.choose_draft_pass(state, remaining)

        # ожидаем: передастся худшая для соперника карта, то есть 6A (index 1)
        self.assertEqual(pass_idx, 1)

    def test_pass_avoids_completing_opponent_I_reversed_order(self) -> None:
        cfg = load_rules_config(_cfg_path())
        state = create_initial_game_state(cfg, num_players=2, rng=random.Random(1), game_id=1)
        state.current_player_idx = 0

        b = state.players[1].board
        b.place(0, 0, Card(rank=1, color=1, shape=1, uid="t1"))
        b.place(1, 0, Card(rank=2, color=2, shape=1, uid="t2"))
        b.place(2, 0, Card(rank=3, color=3, shape=1, uid="t3"))

        remaining = [
            Card(rank=6, color=1, shape=1, uid="bad_6A"),
            Card(rank=4, color=4, shape=1, uid="good_4D"),
        ]

        agent = Line4HAgent(random.Random(0), cfg)
        pass_idx = agent.choose_draft_pass(state, remaining)

        # теперь “мусор” на позиции 0
        self.assertEqual(pass_idx, 0)


if __name__ == "__main__":
    unittest.main()
