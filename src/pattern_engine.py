from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config_loader import PatternSpec, RulesConfig
from models import Board, Card, Coord, Joker, PatternMatch, PlacedCard


@dataclass(frozen=True, slots=True)
class _Window:
    coords: Tuple[Coord, ...]
    shape: str
    size: int


def find_patterns_on_board(board: Board, cfg: RulesConfig) -> List[PatternMatch]:
    """
    Ищет все совпадения паттернов из cfg.patterns на указанной доске.
    Возвращает список PatternMatch, каждый содержит:
      - pattern_id
      - cells (координаты карт паттерна)
      - vp
      - reward_id (или None)
    """
    ranks_domain = tuple(cfg.game.deck.ranks)
    colors_domain = tuple(cfg.game.deck.colors)
    joker_enabled = cfg.game.joker.enabled

    windows_by_shape = _precompute_windows(board.width, board.height)

    out: List[PatternMatch] = []
    for pat in cfg.patterns:
        shape_windows = windows_by_shape.get(pat.shape, [])
        for w in shape_windows:
            if w.size != pat.size_cards:
                continue

            placed = _get_window_cards(board, w.coords)
            if placed is None:
                continue

            if not joker_enabled and any(isinstance(c, Joker) for c in placed):
                continue

            if not _match_rank_rule(placed, w, pat.rank_rule, ranks_domain):
                continue
            if not _match_color_rule(placed, w, pat.color_rule, colors_domain):
                continue

            out.append(
                PatternMatch(
                    pattern_id=pat.id,
                    cells=w.coords,
                    vp=pat.vp,
                    reward_id=(pat.reward.id if pat.reward is not None else None),
                )
            )

    return out


def _get_window_cards(board: Board, coords: Tuple[Coord, ...]) -> Optional[Tuple[PlacedCard, ...]]:
    cards: List[PlacedCard] = []
    for (x, y) in coords:
        c = board.get(x, y)
        if c is None:
            return None
        cards.append(c)
    return tuple(cards)


@lru_cache(maxsize=64)
def _precompute_windows(width: int, height: int) -> Dict[str, List[_Window]]:
    out: Dict[str, List[_Window]] = {
        "square_3x3": [],
        "line_len_5": [],
        "line_len_4": [],
    }

    # square_3x3: все окна 3x3, coords идут row-major: (x0,y0)..(x0+2,y0+2)
    if width >= 3 and height >= 3:
        for y0 in range(0, height - 3 + 1):
            for x0 in range(0, width - 3 + 1):
                coords = tuple((x0 + dx, y0 + dy) for dy in range(3) for dx in range(3))
                out["square_3x3"].append(_Window(coords=coords, shape="square_3x3", size=9))

    # line_len_5: горизонтали и вертикали длины 5, порядок слева-направо и сверху-вниз
    if width >= 5:
        for y in range(height):
            coords = tuple((x, y) for x in range(0, 5))
            out["line_len_5"].append(_Window(coords=coords, shape="line_len_5", size=5))
    if height >= 5:
        for x in range(width):
            coords = tuple((x, y) for y in range(0, 5))
            out["line_len_5"].append(_Window(coords=coords, shape="line_len_5", size=5))

    # line_len_4: все подотрезки длины 4 по горизонтали и вертикали
    if width >= 4:
        for y in range(height):
            for x0 in range(0, width - 4 + 1):
                coords = tuple((x0 + dx, y) for dx in range(4))
                out["line_len_4"].append(_Window(coords=coords, shape="line_len_4", size=4))
    if height >= 4:
        for x in range(width):
            for y0 in range(0, height - 4 + 1):
                coords = tuple((x, y0 + dy) for dy in range(4))
                out["line_len_4"].append(_Window(coords=coords, shape="line_len_4", size=4))

    return out


def _ranks_and_colors(placed: Sequence[PlacedCard]) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    ranks: List[Optional[int]] = []
    colors: List[Optional[int]] = []
    for c in placed:
        if isinstance(c, Card):
            ranks.append(c.rank)
            colors.append(c.color)
        elif isinstance(c, Joker):
            ranks.append(None)
            colors.append(None)
        else:
            raise TypeError(f"Unexpected placed card type: {type(c)}")
    return ranks, colors


def _match_rank_rule(
    placed: Sequence[PlacedCard],
    window: _Window,
    rank_rule: str,
    ranks_domain: Sequence[int],
) -> bool:
    ranks, _ = _ranks_and_colors(placed)
    domain_set = set(ranks_domain)

    if rank_rule == "R1_line_sequence_step1":
        if window.shape not in ("line_len_5", "line_len_4"):
            return False
        return _rank_line_sequence_step1(ranks, domain_set)

    if rank_rule == "R2_line_all_equal":
        if window.shape not in ("line_len_5", "line_len_4"):
            return False
        return _rank_all_equal(ranks)

    if rank_rule == "R3_square_stripes":
        if window.shape != "square_3x3":
            return False
        return _rank_square_stripes(ranks, domain_set)

    if rank_rule == "R4_square_two_axes":
        if window.shape != "square_3x3":
            return False
        return _rank_square_two_axes(ranks, domain_set)

    raise ValueError(f"Unsupported rank_rule: {rank_rule}")


def _match_color_rule(
    placed: Sequence[PlacedCard],
    window: _Window,
    color_rule: str,
    colors_domain: Sequence[int],
) -> bool:
    _, colors = _ranks_and_colors(placed)
    domain_set = set(colors_domain)

    if color_rule == "C1_line_monochrome":
        if window.shape not in ("line_len_5", "line_len_4"):
            return False
        return _color_all_equal(colors)

    if color_rule == "C2_line_all_distinct":
        if window.shape not in ("line_len_5", "line_len_4"):
            return False
        return _color_all_distinct(colors, domain_set)

    if color_rule == "C3_square_flag":
        if window.shape != "square_3x3":
            return False
        return _color_square_flag(colors, domain_set)

    if color_rule == "C4_square_latin":
        if window.shape != "square_3x3":
            return False
        return _color_square_latin(colors, domain_set)

    raise ValueError(f"Unsupported color_rule: {color_rule}")


def _rank_line_sequence_step1(ranks: Sequence[Optional[int]], domain: Set[int]) -> bool:
    """
    Есть ли последовательность длины L с шагом 1, которая совместима с фиксированными значениями.
    Допускаем и возрастание, и убывание.
    """
    L = len(ranks)
    for start in domain:
        for direction in (1, -1):
            ok = True
            for i in range(L):
                expected = start + direction * i
                if expected not in domain:
                    ok = False
                    break
                fixed = ranks[i]
                if fixed is not None and fixed != expected:
                    ok = False
                    break
            if ok:
                return True
    return False


def _rank_all_equal(ranks: Sequence[Optional[int]]) -> bool:
    fixed_vals = [r for r in ranks if r is not None]
    if not fixed_vals:
        return True
    return len(set(fixed_vals)) == 1


def _rank_square_stripes(ranks_row_major: Sequence[Optional[int]], domain: Set[int]) -> bool:
    """
    3x3, stripes: либо строки константны и значения строк идут подряд шаг=1,
    либо столбцы константны и значения столбцов идут подряд шаг=1.
    """
    grid = _to_3x3(ranks_row_major)

    # Rows orientation
    if _rank_stripes_rows(grid, domain):
        return True
    # Cols orientation
    if _rank_stripes_cols(grid, domain):
        return True

    return False


def _rank_stripes_rows(grid: List[List[Optional[int]]], domain: Set[int]) -> bool:
    # grid[y][x]
    for start in domain:
        for direction in (1, -1):
            row_vals = [start + direction * i for i in range(3)]
            if any(v not in domain for v in row_vals):
                continue

            ok = True
            for y in range(3):
                expected = row_vals[y]
                for x in range(3):
                    fixed = grid[y][x]
                    if fixed is not None and fixed != expected:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _rank_stripes_cols(grid: List[List[Optional[int]]], domain: Set[int]) -> bool:
    for start in domain:
        for direction in (1, -1):
            col_vals = [start + direction * i for i in range(3)]
            if any(v not in domain for v in col_vals):
                continue

            ok = True
            for x in range(3):
                expected = col_vals[x]
                for y in range(3):
                    fixed = grid[y][x]
                    if fixed is not None and fixed != expected:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _rank_square_two_axes(ranks_row_major: Sequence[Optional[int]], domain: Set[int]) -> bool:
    """
    3x3: rank(x,y) = base + dx*x + dy*y, dx,dy in {+1,-1}
    """
    grid = _to_3x3(ranks_row_major)
    for base in domain:
        for dx in (1, -1):
            for dy in (1, -1):
                ok = True
                for y in range(3):
                    for x in range(3):
                        expected = base + dx * x + dy * y
                        if expected not in domain:
                            ok = False
                            break
                        fixed = grid[y][x]
                        if fixed is not None and fixed != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return True
    return False


def _color_all_equal(colors: Sequence[Optional[int]]) -> bool:
    fixed_vals = [c for c in colors if c is not None]
    if not fixed_vals:
        return True
    return len(set(fixed_vals)) == 1


def _color_all_distinct(colors: Sequence[Optional[int]], domain: Set[int]) -> bool:
    L = len(colors)
    if len(domain) < L:
        return False

    fixed = [c for c in colors if c is not None]
    if len(set(fixed)) != len(fixed):
        return False

    jokers = sum(1 for c in colors if c is None)
    available = domain - set(fixed)
    return len(available) >= jokers


def _color_square_flag(colors_row_major: Sequence[Optional[int]], domain: Set[int]) -> bool:
    """
    3x3: либо каждая строка монохромна, и 3 строки разного цвета,
    либо каждый столбец монохромен, и 3 столбца разного цвета.
    """
    grid = _to_3x3(colors_row_major)

    if _color_flag_rows(grid, domain):
        return True
    if _color_flag_cols(grid, domain):
        return True
    return False


def _color_flag_rows(grid: List[List[Optional[int]]], domain: Set[int]) -> bool:
    # Для каждой строки: либо фикс. цвет (если есть), либо None.
    required: List[Optional[int]] = []
    for y in range(3):
        s = {grid[y][x] for x in range(3) if grid[y][x] is not None}
        if len(s) > 1:
            return False
        required.append(next(iter(s)) if s else None)

    for cols in permutations(domain, 3):
        ok = True
        for y in range(3):
            assigned = cols[y]
            if required[y] is not None and required[y] != assigned:
                ok = False
                break
            # если в строке есть фиксированные клетки, они уже согласованы required
        if ok:
            return True
    return False


def _color_flag_cols(grid: List[List[Optional[int]]], domain: Set[int]) -> bool:
    required: List[Optional[int]] = []
    for x in range(3):
        s = {grid[y][x] for y in range(3) if grid[y][x] is not None}
        if len(s) > 1:
            return False
        required.append(next(iter(s)) if s else None)

    for cols in permutations(domain, 3):
        ok = True
        for x in range(3):
            assigned = cols[x]
            if required[x] is not None and required[x] != assigned:
                ok = False
                break
        if ok:
            return True
    return False


def _color_square_latin(colors_row_major: Sequence[Optional[int]], domain: Set[int]) -> bool:
    """
    3x3: в каждой строке все цвета различны, и в каждом столбце все цвета различны.
    Домена 5 цветов достаточно, это решаем через backtracking.
    """
    grid = _to_3x3(colors_row_major)

    row_used: List[Set[int]] = [set() for _ in range(3)]
    col_used: List[Set[int]] = [set() for _ in range(3)]

    for y in range(3):
        for x in range(3):
            v = grid[y][x]
            if v is None:
                continue
            if v not in domain:
                return False
            if v in row_used[y]:
                return False
            if v in col_used[x]:
                return False
            row_used[y].add(v)
            col_used[x].add(v)

    empties: List[Tuple[int, int]] = [(x, y) for y in range(3) for x in range(3) if grid[y][x] is None]

    def pick_next_cell() -> int:
        best_i = 0
        best_len = 10**9
        for i, (x, y) in enumerate(empties):
            choices = domain - row_used[y] - col_used[x]
            if len(choices) < best_len:
                best_len = len(choices)
                best_i = i
                if best_len <= 1:
                    break
        return best_i

    def backtrack() -> bool:
        if not empties:
            return True

        i = pick_next_cell()
        x, y = empties.pop(i)

        choices = list(domain - row_used[y] - col_used[x])
        if not choices:
            empties.insert(i, (x, y))
            return False

        for v in choices:
            grid[y][x] = v
            row_used[y].add(v)
            col_used[x].add(v)

            if backtrack():
                return True

            row_used[y].remove(v)
            col_used[x].remove(v)
            grid[y][x] = None

        empties.insert(i, (x, y))
        return False

    return backtrack()


def _to_3x3(values_row_major: Sequence[Optional[int]]) -> List[List[Optional[int]]]:
    if len(values_row_major) != 9:
        raise ValueError(f"Expected 9 values for 3x3, got {len(values_row_major)}")
    return [
        [values_row_major[0], values_row_major[1], values_row_major[2]],
        [values_row_major[3], values_row_major[4], values_row_major[5]],
        [values_row_major[6], values_row_major[7], values_row_major[8]],
    ]
