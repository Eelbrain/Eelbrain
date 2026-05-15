from contextlib import ContextDecorator

from appnope import NSActivityUserInitiated, beginActivityWithOptions, endActivity


class ActivityContext(ContextDecorator):
    """Context disabling idle sleep and App Nap on macOS"""

    def __init__(self, options, message):
        self.n_processes = 0
        self.options = options
        self.message = message

    def __enter__(self):
        if self.n_processes == 0:
            self._activity = beginActivityWithOptions(self.options, self.message)
        self.n_processes += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.n_processes -= 1
        if self.n_processes == 0:
            endActivity(self._activity)


user_activity = ActivityContext(NSActivityUserInitiated, 'Eelbrain user activity')
