"""Plot non-uniform time series"""


class PltBinNuts:
    def __init__(self, ax, epoch, color='r', fill=False, hatch='//', **kwargs):
        """Plot a simple on/off nonuniform time series

        Parameters
        ----------
        ax : matplotlib axes
            Target axes.
        epoch : array
            Array with fields 'start' and 'stop'.
        kwargs :
            axvspan keyword arguments.
        """
        self._handles = []
        for line in epoch:
            start = line['start']
            stop = line['stop']
            h = ax.axvspan(start, stop, color=color, fill=fill, hatch=hatch, **kwargs)
            self._handles.append(h)
