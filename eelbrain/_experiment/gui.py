"""CLI entry points for eelbrain."""
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``eelbrain-gui`` command.

    Parameters
    ----------
    argv
        Command-line arguments. If omitted, arguments are read from
        ``sys.argv``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='eelbrain-gui',
        description='Open the Eelbrain pipeline GUI',
    )
    parser.add_argument(
        'path', nargs='?', default=None,
        help='Pipeline file or directory (default: current working directory)',
    )
    parser.add_argument(
        '--log-level',
        dest='log_level',
        help='Determine log level for log messages printed to the terminal; overrides Pipeline.screen_log_level for the loaded pipeline',
    )
    args = parser.parse_args(argv)

    from .load_pipeline import load_pipeline
    from .._wxgui.app import get_app
    from .._wxgui.pipeline_gui import PipelineFrame

    pipeline = load_pipeline(args.path, log_level=args.log_level)
    app = get_app(jumpstart=True)
    PipelineFrame(pipeline).Show()
    app.MainLoop()
