"""Shared grouping helpers for the detection stages."""

from __future__ import annotations


class UnionFind:
    """Union-find with path compression; merges pairwise matches into groups."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def clusters(self) -> list[list[int]]:
        """Groups of size >= 2, members sorted, deterministic order."""
        by_root: dict[int, list[int]] = {}
        for x in self._parent:
            by_root.setdefault(self.find(x), []).append(x)
        out = [sorted(members) for members in by_root.values() if len(members) > 1]
        out.sort()
        return out
