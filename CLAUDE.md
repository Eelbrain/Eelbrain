# Code formatting

- Use f-strings rather than older string formatting methods (except for logging). 
  Use "{x=}" instead of "x={x!r}".
- Keep long string literals (and functions thereof, such as error messages) on one line (ignore line length limit)
- Use type hints in function signatures consistently
- Document all arguments in public functions and methods in numpydoc style
  (and don't duplicate the type hints in the docstring)
- When function signatures have two or more arguments with non-trivial types, make it one line per argument
- Keep code concise and linear, but maximize readability
  - Prefer an explicit if/else statement over a conditional expression (`x if c else y`) when the branches are more than trivial
  - Avoid heavily nested one-liners
  - Use meaningful variable names

# Environment

The development and testing environment is `env-dev.yml`.
It is usually an existing `mamba` environment called `eeldev`.
