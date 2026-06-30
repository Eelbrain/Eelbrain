"""Set eelbrain.__version__ from the current GitHub release tag."""
import os
import re
from pathlib import Path


VERSION_RE = re.compile(r"\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?")
VERSION_ASSIGNMENT_RE = re.compile(r"^__version__ = ['\"][^'\"]+['\"]$", re.MULTILINE)


def main() -> None:
    tag = os.environ["GITHUB_REF_NAME"]
    version = tag.removeprefix("v")
    if not VERSION_RE.fullmatch(version):
        raise SystemExit(f"Unsupported release tag {tag!r}; expected v<PEP 440 version>")

    repo = Path(__file__).resolve().parents[2]
    init_path = repo / "eelbrain" / "__init__.py"
    text = init_path.read_text()
    text, count = VERSION_ASSIGNMENT_RE.subn(f"__version__ = {version!r}", text)
    if count != 1:
        raise SystemExit("Could not replace exactly one __version__ assignment")

    init_path.write_text(text)
    print(f"Set eelbrain.__version__ to {version}")


if __name__ == "__main__":
    main()
