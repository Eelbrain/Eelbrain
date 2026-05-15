from contextlib import ContextDecorator


class NullContext(ContextDecorator):
    """Context disabling idle sleep and App Nap on macOS"""

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


user_activity = NullContext()
