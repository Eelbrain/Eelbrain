"""Use r (rpy2) for testing"""
import warnings
from rpy2.robjects import r

try:
    from rpy2.rinterface import RRuntimeWarning
except ImportError:  # rpy2 < 2.8
    RRuntimeWarning = UserWarning


def r_require(package):
    with r_warning_filter:
        res = r('require(%s)' % package)
        success = res[0] if res is not None else None

    if not success:
        print(r("install.packages('%s', repos='http://cran.us.r-project.org')"
                % package))
        res = r('require(%s)' % package)
        success = res[0] if res is not None else None
        if not success:
            raise RuntimeError("Could not install R package %r" % package)


class RWarningFilter:

    def __enter__(self):
        self.context = warnings.catch_warnings()
        self.context.__enter__()
        warnings.filterwarnings('ignore', category=RRuntimeWarning)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.__exit__(exc_type, exc_val, exc_tb)


r_warning_filter = RWarningFilter()
