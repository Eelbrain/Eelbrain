# AGENTS

## Code Style

- Be concise and make sure there is a single source of truth. 
- Avoid multiple ways of getting the same object and pass-through functions.
- Avoid defining new methods for single use.

## Sandbox Notes

- If Python code using `multiprocessing.shared_memory` fails with `PermissionError: [Errno 1] Operation not permitted` while creating a `/psm_*` segment, rerun that command outside the sandbox with escalated permissions.
- This mainly matters for permutation/statistics code that uses POSIX shared memory; changing writable directories does not fix it.

## Verification

- Prefer targeted verification for the files and behavior you changed.
- To verify code style after making changes, run `pre-commit run --files <changed files>`.
- Run the smallest relevant pytest target.

## Formatting

- Follow the surrounding style of the file you edit.
- There is no hard line-length limit here; keep readable expressions on one line when that is clearer.
- Reflow code only when it improves readability.
- Avoid unrelated formatting cleanup in files you are not otherwise changing.
