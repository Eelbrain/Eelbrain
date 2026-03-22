# AGENTS

## Sandbox Notes

- If Python code using `multiprocessing.shared_memory` fails with `PermissionError: [Errno 1] Operation not permitted` while creating a `/psm_*` segment, rerun that command outside the sandbox with escalated permissions.
- Changing writable directories or ordinary filesystem sandbox settings does not fix this issue, because `multiprocessing.shared_memory` relies on POSIX shared memory (`shm_open`), which is blocked by the sandbox.

## Verification

- Use the repo's commit hooks to verify coding style before finishing a change, for example with `pre-commit run --all-files`.

## Formatting

- Prefer longer readable lines over fractionating expressions into many short lines (we don't impose a maximum line length, but rely on soft wrapping instead). For example, keep chained path-building expressions and straightforward function and method calls on one line.
- Distribute commands across multiple lines only when it increases readability (for example, arguments to a function, each with its own type hint and default value, and large dictionary initialization may be clearer when each is on its own line).
