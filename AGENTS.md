# AGENTS

## Code Structure

- Prefer direct implementations with one obvious source of truth.
- Avoid thin pass-through helpers that only rename or forward existing behavior.
- Do not add one-off helpers unless they materially improve readability, reuse, or testability.

## Sandbox Notes

- If Python code using `multiprocessing.shared_memory` fails with `PermissionError: [Errno 1] Operation not permitted` while creating a `/psm_*` segment, rerun that command outside the sandbox with escalated permissions.
- This mainly matters for permutation/statistics code that uses POSIX shared memory; changing writable directories does not fix it.

## Verification

- Prefer targeted verification for the files and behavior you changed.
- Always verify code style after making changes, run `pre-commit run --files <changed files>`.
- Run the smallest relevant pytest target first.
- If no narrower target is obvious, use `make test-no-gui` for non-GUI changes.
- For GUI changes, use the `pythonw`-based targets in `Makefile`.

## Formatting

- Follow the surrounding style of the file you edit.
- There is no hard line-length limit here; keep readable expressions on one line when that is clearer.
- For long strings, prefer one readable long literal over splitting the text into adjacent substrings just to wrap lines.
- Reflow code only when it improves readability.
- Avoid unrelated formatting cleanup in files you are not otherwise changing.
