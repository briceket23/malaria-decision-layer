"""Failures that tell the operator what to do next.

Every error raised by this package carries a remedy. The rule is not politeness:
during the predecessor's development an hour went to a crash whose message was
`ModuleNotFoundError: No module named 'models'`, which named neither the cause
(a repository-format checkpoint) nor the fix (install the packaged fork, or
point at a clone). A message that states only what failed pushes the diagnosis
onto whoever is least equipped to do it.

`MagError.__str__` prints the problem, then what to do, then the context, in
that order, because that is the order in which they are useful.
"""

from __future__ import annotations


class MagError(Exception):
    """Base class. Always constructed with a remedy."""

    def __init__(self, problem: str, remedy: str = "", **context):
        self.problem = problem
        self.remedy = remedy
        self.context = context
        super().__init__(problem)

    def __str__(self) -> str:
        out = [self.problem]
        if self.remedy:
            out += ["", "What to do:", *(f"  {ln}" for ln in self.remedy.splitlines())]
        if self.context:
            out += ["", "Context:"]
            out += [f"  {k} = {v!r}" for k, v in sorted(self.context.items())]
        return "\n".join(out)


class MissingInput(MagError):
    """A file or directory the analysis needs is not there."""


class BadInput(MagError):
    """An input exists but is not what it claims to be."""


class IntegrityError(MagError):
    """A cache no longer matches the manifest written beside it."""


class ProvenanceError(MagError):
    """Results were asked for that cannot be traced to a run on the archive."""


class GateRefusal(MagError):
    """A gate refused to let the analysis continue.

    This is not a malfunction and must never be caught and ignored. The gates
    exist to stop three specific things: analysing a pipeline that does not
    reproduce its own baseline, testing a comparison that cannot produce a
    non-null result, and treating a detector that did not detect as a weak
    detector.
    """
