# skip test: data unavailable
from pathlib import Path


def iter_top_subject_dirs(root: Path):
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("sub-"):
            yield p


def plan_bem_file_renames(subj_dir: Path):
    bem = subj_dir / "bem"
    if not bem.is_dir():
        return
    for f in sorted(bem.iterdir()):
        if f.is_file() and f.name.startswith("sub-"):
            yield f, f.with_name(f.name[4:])


def plan_dir_rename(subj_dir: Path):
    name = subj_dir.name
    if name.startswith("sub-"):
        target = subj_dir.with_name(name[4:])
        return subj_dir, target
    return None


def main(root: str | Path, dry_run: bool = False):
    """Run the rename routine (see module docstring)."""
    root_path = Path(root).expanduser().resolve()

    if not root_path.is_dir():
        print(f"[ERROR] Root directory is not accessible: {root_path}")
        return

    bem_plans = []
    dir_plans = []

    # Stage 1: BEM files
    for subj in iter_top_subject_dirs(root_path):
        for old, new in plan_bem_file_renames(subj):
            if not new.exists():
                bem_plans.append((old, new))
            else:
                print(f"[SKIP] Conflict (file exists): {new}")

    # Stage 2: Directories
    for subj in iter_top_subject_dirs(root_path):
        plan = plan_dir_rename(subj)
        if plan:
            old, new = plan
            if not new.exists():
                dir_plans.append((old, new))
            else:
                print(f"[SKIP] Conflict (directory exists): {new}")

    if not bem_plans and not dir_plans:
        print("No items to rename.")
        return 0

    # Plan output
    if bem_plans:
        print("=== BEM file rename plan ===")
        for o, n in bem_plans:
            print(f"FILE: {o.name} -> {n.name}")
    if dir_plans:
        print("=== Directory rename plan (after BEM files) ===")
        for o, n in dir_plans:
            print(f"DIR:  {o.name} -> {n.name}")
    if dry_run:
        print("\nDry-run mode: no changes made.")
        return

    # Execute: files first (BEM), then directories
    for o, n in bem_plans:
        try:
            o.rename(n)
            print(f"[OK] FILE {o.name} -> {n.name}")
        except Exception as e:
            print(f"[ERR] {o} -> {n}: {e}")

    for o, n in dir_plans:
        try:
            o.rename(n)
            print(f"[OK] DIR  {o.name} -> {n.name}")
        except Exception as e:
            print(f"[ERR] {o} -> {n}: {e}")


if __name__ == "__main__":
    # Set dry_run=True to only show the planned rename operations without making changes.
    main('/mnt/d/Data/ds005810/derivatives/freesurfer', dry_run=False)
