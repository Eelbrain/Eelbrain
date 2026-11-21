"""CLI entry points for eelbrain."""
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``eelbrain`` command.

    Parameters
    ----------
    argv
        Command-line arguments. If omitted, arguments are read from
        ``sys.argv``.
    """
    import argparse

    from .. import __version__

    parser = argparse.ArgumentParser(
        prog='eelbrain',
        description='Open the Eelbrain pipeline GUI',
    )
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument(
        'path', nargs='?', default=None,
        help='Pipeline file or directory (default: current working directory)',
    )
    parser.add_argument(
        '--log-level',
        dest='log_level',
        help='Determine log level for log messages printed to the terminal; overrides Pipeline.screen_log_level for the loaded pipeline',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print debug output at --log-level debug, and log how long each step of a table refresh takes and which node each artifact load spends its time in',
    )
    args = parser.parse_args(argv)

    from .load_pipeline import load_pipeline

    pipeline = load_pipeline(args.path, log_level=args.log_level or ('debug' if args.debug else None))
    if args.debug:
        # attribute a slow table row to the node whose load took the time
        pipeline._derivatives.profile_loads = True

    from .._wxgui.app import get_app
    from .._wxgui.pipeline_gui import PipelineFrame

    app = get_app(jumpstart=True)
    PipelineFrame(pipeline).Show()
    app.MainLoop()
