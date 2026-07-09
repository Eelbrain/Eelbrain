# Code formatting

- Use f-strings rather than older string formatting methods. Use "{x=}" instead of "x={x!r}".
- Whenever possible, keep long string literals (and functions thereof, such as error messages) on one line (ignore line length limit)
- Use type hints in function signatures consistently
- Document all arguments in numpydoc style  (and don't duplicate the type hints in the docstring)
- When function signatures have two or more arguments with non-trivial types, make it one line per argument


# Environment

The development and testing environment is `env-dev.yml`.
It is usually an existing `mamba` environment called `eeldev`.
