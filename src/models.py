from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union


Coord = Tuple[int, int]


@dataclass(frozen=True, slots=True)
class Card:
    rank: int
    color: int
    shape: int
    uid: str


@dataclass(frozen=True, slots=True)
class Joker:
    uid: str = "JOKER"


PlacedCard = Union[Card, Joker]


@dataclass(slots=True)
class Board:
    width: int
    height: int
    _cells: List[List[Optional[PlacedCard]]] = field(init=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Board width/height must be > 0")
        self._cells = [[None for _ in range(self.width)] for _ in range(self.height)]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> Optional[PlacedCard]:
        if not self.in_bounds(x, y):
            raise IndexError(f"Out of bounds cell ({x},{y})")
        return self._cells[y][x]

    def is_empty(self, x: int, y: int) -> bool:
        return self.get(x, y) is None

    def set(self, x: int, y: int, card: Optional[PlacedCard]) -> None:
        if not self.in_bounds(x, y):
            raise IndexError(f"Out of bounds cell ({x},{y})")
        self._cells[y][x] = card

    def place(self, x: int, y: int, card: PlacedCard) -> None:
        if not self.in_bounds(x, y):
            raise IndexError(f"Out of bounds cell ({x},{y})")
        if self._cells[y][x] is not None:
            raise ValueError(f"Cell ({x},{y}) is not empty")
        self._cells[y][x] = card

    def remove(self, x: int, y: int) -> PlacedCard:
        if not self.in_bounds(x, y):
            raise IndexError(f"Out of bounds cell ({x},{y})")
        card = self._cells[y][x]
        if card is None:
            raise ValueError(f"Cell ({x},{y}) is empty")
        self._cells[y][x] = None
        return card

    def move(self, src: Coord, dst: Coord) -> None:
        sx, sy = src
        dx, dy = dst
        if not self.in_bounds(sx, sy) or not self.in_bounds(dx, dy):
            raise IndexError("Out of bounds move")
        if self._cells[sy][sx] is None:
            raise ValueError(f"Source cell {src} is empty")
        if self._cells[dy][dx] is not None:
            raise ValueError(f"Destination cell {dst} is not empty")
        self._cells[dy][dx] = self._cells[sy][sx]
        self._cells[sy][sx] = None

    def empty_cells(self) -> List[Coord]:
        out: List[Coord] = []
        for y in range(self.height):
            for x in range(self.width):
                if self._cells[y][x] is None:
                    out.append((x, y))
        return out

    def has_space(self) -> bool:
        for y in range(self.height):
            for x in range(self.width):
                if self._cells[y][x] is None:
                    return True
        return False

    def occupied_cells(self) -> List[Tuple[Coord, PlacedCard]]:
        out: List[Tuple[Coord, PlacedCard]] = []
        for y in range(self.height):
            for x in range(self.width):
                c = self._cells[y][x]
                if c is not None:
                    out.append(((x, y), c))
        return out

    def snapshot_uids(self) -> Tuple[Tuple[Optional[str], ...], ...]:
        return tuple(
            tuple((c.uid if c is not None else None) for c in row)
            for row in self._cells
        )


@dataclass(frozen=True, slots=True)
class ProtectedZone:
    size: int

    def is_protected(self, x: int, y: int, board_width: int, board_height: int) -> bool:
        if self.size <= 0:
            return False
        if self.size % 2 == 0:
            raise ValueError("Protected zone size must be odd (e.g., 3)")
        if self.size > board_width or self.size > board_height:
            raise ValueError("Protected zone size cannot exceed board dimensions")

        start_x = (board_width - self.size) // 2
        start_y = (board_height - self.size) // 2
        end_x = start_x + self.size - 1
        end_y = start_y + self.size - 1

        return start_x <= x <= end_x and start_y <= y <= end_y


@dataclass(slots=True)
class PlayerState:
    player_idx: int
    board: Board
    hand: List[Card] = field(default_factory=list)
    vp: int = 0
    skip_next_turn: bool = False


EndReason = str  # ожидаем "END1" или "END2"


@dataclass(slots=True)
class GameState:
    players: List[PlayerState]
    current_player_idx: int
    deck: List[Card]
    discard: List[PlacedCard]
    protected_zone: ProtectedZone
    turn_number: int = 0
    ended: bool = False
    end_reason: Optional[EndReason] = None
    game_id: int = 0

    def next_player_idx(self) -> int:
        return (self.current_player_idx + 1) % len(self.players)

    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    def player_left_of(self, player_idx: int) -> int:
        return (player_idx + 1) % len(self.players)

    def is_opponent_cell_protected(self, opponent_idx: int, x: int, y: int) -> bool:
        opp_board = self.players[opponent_idx].board
        return self.protected_zone.is_protected(x, y, opp_board.width, opp_board.height)


@dataclass(frozen=True, slots=True)
class PatternMatch:
    pattern_id: str
    cells: Tuple[Coord, ...]
    vp: int
    reward_id: Optional[str]


@dataclass(frozen=True, slots=True)
class FallbackEvent:
    game_id: int
    turn_number: int
    player_idx: int
    agent_name: str
    decision_type: str
    reason: str
    chosen: str


@dataclass(slots=True)
class MatchAndRewardResult:
    player_idx: int
    pattern_id: str
    gained_vp: int
    reward_id: Optional[str]
    reward_applied: bool
    reward_refused: bool
    reward_impossible: bool
