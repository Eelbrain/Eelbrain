# AGENTS

## Sandbox Notes

- If Python code using `multiprocessing.shared_memory` fails with `PermissionError: [Errno 1] Operation not permitted` while creating a `/psm_*` segment, rerun that command outside the sandbox with escalated permissions.
- This mainly matters for permutation/statistics code that uses POSIX shared memory; changing writable directories does not fix it.

## Verification

- Prefer targeted verification for the files and behavior you changed.
- For hooks, use `pre-commit run --files <changed files>` rather than `--all-files`.
- Run the smallest relevant pytest target first.
- If you touch shared non-GUI behavior, finish with `make test-no-gui`.
- If you touch GUI code, use the `pythonw`-based targets in `Makefile`.
- If you change docstrings or public API docs, run `pydocstyle eelbrain`.

## Formatting

- Follow the surrounding style of the file you edit.
- There is no hard line-length limit here; keep readable expressions on one line when that is clearer.
- Reflow code only when it improves readability.
- Avoid unrelated formatting cleanup in files you are not otherwise changing.
