import cProfile
from optparse import OptionParser
import pstats

import mne
from eelbrain import *

mne.set_log_level('warning')
configure(n_workers=False)


# option parser
parser = OptionParser()
parser.add_option("-m", "--make", dest="make", metavar="KIND",
                  help="Make a new profile of kind mne or uts")
parser.add_option("-f", "--file", dest="file_ext", metavar="NAME",
                  help="Use a different file for this profile")
parser.add_option("-s", "--sort", dest="sort", metavar="CRITERION",
                  help="Sort the profile entries according to CRITERION")
parser.add_option("-n", dest="number", metavar="NUMBER",
                  help="Display NUMBER entries from the profile.")
(options, args) = parser.parse_args()


# process options
if options.file_ext is None:
    fname = 'profile_of_anova.profile'
else:
    fname = f'profile_of_anova_{options.file_ext}.profile'

sort = options.sort
if options.number is None:
    number = 20
    if sort is None:
        sort = 'tottime'
else:
    number = int(options.number)


# run profile
make = options.make
if make:
    if make == 'mne':
        ds = datasets.get_mne_sample(-0.1, 0.2, src='ico', sub=None, rm=True)
        statement = ("testnd.ANOVA('src', 'side * modality * subject', "
                     "match='subject', ds=ds, samples=2, tstart=0)")
    elif make == 'uts':
        ds = datasets.get_rand(True)
        statement = ("testnd.ANOVA('uts', 'A * B * rm', ds=ds, samples=100, "
                     "tfce=True, tstart=0, match='rm')")
    else:
        raise ValueError(f"-m {make}")
    cProfile.run(statement, fname)


# display stats
p = pstats.Stats(fname)
if sort:
    p.sort_stats(sort)
p.print_stats(number)
