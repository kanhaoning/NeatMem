"""NeatMem exception types."""


class NeatMemValidationError(ValueError):
    """Validation error with message text aligned to mem0's Mem0ValidationError.

    Inherits ValueError so existing ``except ValueError`` handlers keep working;
    catching this specific type requires ``from neatmem import NeatMemValidationError``.
    """
