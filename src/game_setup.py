from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from config_loader import RulesConfig
from models import Board, Card, GameState, PlayerState, ProtectedZone


def build_full_unique_deck(cfg: RulesConfig) -> List[Card]:
    deck: List[Card] = []

    ranks = cfg.game.deck.ranks
    colors = cfg.game.deck.colors
    shapes = cfg.game.deck.shapes

    for r in ranks:
        for c in colors:
            for s in shapes:
                uid = f"r{r}c{c}s{s}"
                deck.append(Card(rank=r, color=c, shape=s, uid=uid))

    if cfg.game.deck.unique_cards:
        expected = len(ranks) * len(colors) * len(shapes)
        if len(deck) != expected:
            raise ValueError(f"Deck size mismatch: built={len(deck)} expected={expected}")

    if cfg.game.deck.total_cards != len(deck):
        raise ValueError(
            f"Config deck.total_cards={cfg.game.deck.total_cards} does not match built deck size={len(deck)}"
        )

    uid_set = {c.uid for c in deck}
    if len(uid_set) != len(deck):
        raise ValueError("Deck contains duplicate uids")

    return deck


def _validate_grid_and_protected_zone(cfg: RulesConfig) -> ProtectedZone:
    g = cfg.game.grid
    pz = g.protected_zone

    if pz.position != "center":
        raise ValueError(f"Unsupported protected_zone.position: {pz.position}")
    if pz.shape != "square":
        raise ValueError(f"Unsupported protected_zone.shape: {pz.shape}")
    if pz.size % 2 == 0:
        raise ValueError("protected_zone.size must be odd (e.g., 3)")

    if pz.size > g.width or pz.size > g.height:
        raise ValueError("protected_zone.size cannot exceed grid dimensions")

    return ProtectedZone(size=pz.size)


def create_initial_game_state(
    cfg: RulesConfig,
    num_players: int,
    rng: random.Random,
    game_id: int = 0,
) -> GameState:
    if num_players < cfg.game.players.min_players or num_players > cfg.game.players.max_players:
        raise ValueError(
            f"num_players must be in [{cfg.game.players.min_players}..{cfg.game.players.max_players}], got {num_players}"
        )

    protected_zone = _validate_grid_and_protected_zone(cfg)

    deck = build_full_unique_deck(cfg)
    rng.shuffle(deck)

    players: List[PlayerState] = []
    for i in range(num_players):
        board = Board(cfg.game.grid.width, cfg.game.grid.height)
        players.append(PlayerState(player_idx=i, board=board))

    # Важное соглашение: верх сброса это конец списка: discard[-1]
    discard: List[object] = []

    return GameState(
        players=players,
        current_player_idx=0,
        deck=deck,
        discard=discard,  # type: ignore[arg-type]
        protected_zone=protected_zone,
        turn_number=0,
        ended=False,
        end_reason=None,
        game_id=game_id,
    )
