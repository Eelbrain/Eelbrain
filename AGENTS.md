# AGENTS

## Sandbox Notes

- If Python code using `multiprocessing.shared_memory` fails with `PermissionError: [Errno 1] Operation not permitted` while creating a `/psm_*` segment, rerun that command outside the sandbox with escalated permissions.
- Changing writable directories or ordinary filesystem sandbox settings does not fix this issue, because `multiprocessing.shared_memory` relies on POSIX shared memory (`shm_open`), which is blocked by the sandbox.

## Verification

- Use the repo's commit hooks to verify coding style before finishing a change, for example with `pre-commit run --all-files`.

## Formatting

- Prefer longer readable lines over fractionating expressions into many short lines.
- In particular, keep chained path-building expressions on one line when they remain clear.

Preferred:

```python
return str(Path(self.get('cache-dir')) / 'manifests' / 'annot' / mrisubject / f'{parc}{MANIFEST_SUFFIX}')
```

Avoid:

```python
return str(
    Path(self.get('cache-dir'))
    / 'manifests'
    / 'annot'
    / mrisubject
    / f'{parc}{MANIFEST_SUFFIX}'
)
```
