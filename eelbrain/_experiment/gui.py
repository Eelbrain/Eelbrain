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
    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Migrate legacy derivative files (ICA, trans, bad channels, epoch rejection) to the current BIDS-style layout and exit without opening the GUI',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='With --migrate, explicitly replace files that already exist in the current layout',
    )
    args = parser.parse_args(argv)
    if args.overwrite and not args.migrate:
        parser.error('--overwrite requires --migrate')

    from .load_pipeline import load_pipeline

    pipeline = load_pipeline(args.path, log_level=args.log_level)

    if args.migrate:
        from .migration import migrate_derivatives

        moved = migrate_derivatives(pipeline.root, overwrite=args.overwrite)
        if moved:
            print(f"Migrated {len(moved)} derivative file(s):")
            for old, new in moved:
                print(f"  {old.relative_to(pipeline.root)} -> {new.relative_to(pipeline.root)}")
        else:
            print("No legacy derivative files to migrate.")
        return

    from .._wxgui.app import get_app
    from .._wxgui.pipeline_gui import PipelineFrame

    app = get_app(jumpstart=True)
    PipelineFrame(pipeline).Show()
    app.MainLoop()
