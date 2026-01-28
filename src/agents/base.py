from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from models import Card, Coord, GameState, PatternMatch


DecisionType = str

DECISION_PLACEMENT_CELL: DecisionType = "placement_cell"
DECISION_DRAFT_PICK: DecisionType = "draft_pick"
DECISION_DRAFT_PASS: DecisionType = "draft_pass"
DECISION_PATTERN_TO_RESOLVE: DecisionType = "pattern_to_resolve"
DECISION_APPLY_REWARD: DecisionType = "apply_reward"
DECISION_REWARD_PARAMS: DecisionType = "reward_params"


@dataclass(frozen=True, slots=True)
class RewardContext:
    pattern_id: str
    triggering_cells: Tuple[Coord, ...]
    deck_size: int
    discard_size: int


class Agent(ABC):
    """
    Агент принимает решения, но не имеет права менять state.
    Движок обязан валидировать решения агента и применять fallback,
    если агент вернул невалидное.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def choose_placement_cell(self, state: GameState, card: Card) -> Coord:
        """
        Вернуть координату пустой клетки (x, y) на поле текущего игрока,
        куда будет размещена card.
        """
        raise NotImplementedError

    @abstractmethod
    def choose_draft_pick(self, state: GameState, revealed_cards: Sequence[Card]) -> int:
        """
        revealed_cards содержит 3 карты.
        Нужно вернуть индекс выбранной карты в диапазоне [0..len(revealed_cards)-1].
        """
        raise NotImplementedError

    @abstractmethod
    def choose_draft_pass(self, state: GameState, remaining_cards: Sequence[Card]) -> int:
        """
        remaining_cards содержит 2 карты после выбора pick.
        Нужно вернуть индекс карты для передачи следующему игроку (слева),
        в диапазоне [0..len(remaining_cards)-1].
        Оставшаяся карта уйдет в общий сброс.
        """
        raise NotImplementedError

    @abstractmethod
    def choose_pattern_to_resolve(self, state: GameState, found_patterns: Sequence[PatternMatch]) -> int:
        """
        found_patterns содержит список найденных паттернов на поле текущего игрока.
        Нужно вернуть индекс паттерна, который будет разрешен следующим.
        """
        raise NotImplementedError

    @abstractmethod
    def choose_apply_reward(self, state: GameState, reward_id: str, context: RewardContext) -> bool:
        """
        Вернуть True, если агент хочет попытаться применить награду,
        иначе False (отказ).
        """
        raise NotImplementedError

    @abstractmethod
    def choose_reward_params(self, state: GameState, reward_id: str, context: RewardContext) -> Dict[str, Any]:
        """
        Вернуть параметры награды в виде dict.
        Движок будет валидировать и при необходимости сделает fallback.
        Формат dict зависит от reward_id и будет определен в reward_engine.
        """
        raise NotImplementedError
