# AGENTS

## Sandbox Notes

- If Python code using `multiprocessing.shared_memory` fails with `PermissionError: [Errno 1] Operation not permitted` while creating a `/psm_*` segment, rerun that command outside the sandbox with escalated permissions.
- Changing writable directories or ordinary filesystem sandbox settings does not fix this issue, because `multiprocessing.shared_memory` relies on POSIX shared memory (`shm_open`), which is blocked by the sandbox.
