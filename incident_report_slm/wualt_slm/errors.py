class NotFound(Exception):
    """The requested report does not exist."""


class Conflict(Exception):
    """A request ID was reused with different source data."""
