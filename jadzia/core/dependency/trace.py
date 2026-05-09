"""Tracing classes for dependency management."""

# Standard
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Self

type TraceCallback = Callable[[set[str]], None]
"""Type of callback function that is executed by `Trace.traverse`."""


class Trace:
    """Node in a tree of input data requirements."""

    def __init__(
        self,
        obj: Any,
        *,
        requires: Iterable[str | None] = (),
        inserts: Iterable[str | None] = (),
        removes: Iterable[str | None] = (),
        callback: TraceCallback | None = None,
        parent: Self | None = None,
    ):
        """
        Initialize the node.

        Args:
            obj: Component represented by the current node of the tree.
            requires: Names required by `obj`. These must be provided by another
                upstream component.
            inserts: Names inserted by `obj`. If these are provided by another
                upstream component, they will be overwritten.
            removes: Names removed by `obj`. These columns are not considered
                requirements of `obj`.
            callback: Function that is called once the minimum required names
                are identified for the node by tree traversal.
            parent: Upstream component of `obj` in the data flow.
        """
        if isinstance(obj, str):
            self._objname = obj
        else:
            self._objname = type(obj).__name__

        self._requires = set(filter(None, requires))
        self._inserts = set(filter(None, inserts))
        self._removes = set(filter(None, removes))
        self._names = set()

        self._callback = callback

        self._parent = parent
        self._children: list[Self] = []

        if self._parent is None:
            if self._requires or self._removes:
                raise ValueError("root nodes may not have requirements or removals")
            else:
                self._names.update(self._inserts)
                self._inserts.clear()
        else:
            self._check()
            self._parent._children.append(self)
            self._names.update(self._parent._names)
            self._names.difference_update(self._removes)
            self._names.update(self._inserts)

    def __str__(self) -> str:
        """String representation of a node."""
        return self._objname

    def __len__(self) -> int:
        """
        Size of the traced name set.

        Returns:
            Number of available names.
        """
        return len(self._names)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the traced name set.

        Returns:
            Iterator over available names.
        """
        return iter(self._names)

    def __contains__(self, name: str) -> bool:
        """
        Whether the traced name set contains a given value.

        Args:
            name: Value to check.

        Returns:
            True if the available name set contains `name`.
        """
        return name in self._names

    def update(
        self,
        obj: Any,
        *,
        requires: Iterable[str | None] = (),
        inserts: Iterable[str | None] = (),
        removes: Iterable[str | None] = (),
        callback: TraceCallback | None = None,
    ) -> Self:
        """
        Create, attach, and return a child node.

        Args:
            obj: Component represented by the child node.
            requires: Names required by `obj`.
            inserts: Names inserted by `obj`.
            removes: Names removed by `obj`.
            callback: Function to be called during traversal.

        Returns:
            Newly created child node.
        """
        cls = type(self)
        return cls(
            obj,
            requires=requires,
            inserts=inserts,
            removes=removes,
            callback=callback,
            parent=self,
        )

    def root(self) -> Self:
        """Return the root node of the tree."""
        root = self
        while root._parent is not None:
            root = root._parent

        return root

    def traverse(self) -> set[str]:
        """
        Perform a depth-first traversal of the traced dependency tree.

        Returns:
            Minimum set of required input names for this node.

        Note:
            Traversal is accompanied by the following side effects:

            * Any callback functions that were registered during tracing
              (tree construction) will be called in depth-first order.
            * The node and its children will be disconnected, effectively
              destroying the tree. This is done to assist the garbage
              collector by removing reference cycles.
        """
        if not self._children:
            return self._requires | self._removes

        names = [child.traverse() for child in self._children]
        names = set.union(*names)

        names.difference_update(self._inserts)
        names.update(self._requires)

        if self._callback is not None:
            self._callback(names)

        self._children.clear()
        self._parent = None
        return names

    def _check(self):
        """
        Check that input requirements of a node are satisfied.

        Raises:
            ValueError: If required input names are not available.
        """
        assert self._parent is not None

        required = self._requires | self._removes
        available = self._parent._names
        missing = required - available
        if not missing:
            return

        available_str = ", ".join(sorted(available))
        required_str = ", ".join(sorted(required))
        missing_str = ", ".join(sorted(missing))

        raise ValueError(
            f"failed to trace data flow:\n"
            f"names required by {self}: ({required_str})\n"
            f"names provided by {self._parent}: ({available_str})\n"
            f"missing names: ({missing_str})"
        )


__all__ = ["Trace"]
