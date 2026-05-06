"""CLI entry points for eelbrain."""


def main():
    """Entry point for the ``eelbrain-gui`` command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog='eelbrain-gui',
        description='Open the Eelbrain pipeline GUI',
    )
    parser.add_argument(
        'path', nargs='?', default=None,
        help='Pipeline file or directory (default: current working directory)',
    )
    args = parser.parse_args()

    from .load_pipeline import load_pipeline
    from .._wxgui.app import get_app
    from .._wxgui.pipeline_gui import PipelineFrame

    pipeline = load_pipeline(args.path)
    app = get_app(jumpstart=True)
    PipelineFrame(pipeline).Show()
    app.MainLoop()
