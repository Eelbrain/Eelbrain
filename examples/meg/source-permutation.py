# requires: mayavi
# dataset: mne_sample
"""Create a HTML report file for a permutation test of source space data"""
import eelbrain as e


# settings
n_samples = 1000

#  Load data
ds = e.datasets.get_mne_sample(tmin=-0.1, tmax=0.2, src='ico', sub="modality=='A'")

# compute distribution of max t values through permutation
res = e.testnd.TTestIndependent('src', 'side', 'L', 'R', data=ds, samples=n_samples, tstart=0.05)

# generate parameter map thresholded at p=0.05
pmap = res.masked_parameter_map(pmin=0.05)

# the next line could be used to plot the result for inspection
# (any area that is significant at any time):

# e.plot.brain.cluster(pmap.sum('time'), surf='inflated')

# create an HTML report with the results form the test
report = e.Report("Permutation Test", author="Prof. Enid Gumby")
# add some information about the test
section = report.add_section("Introduction")
text = ("A comparison of auditory stimulation to the left vs. the right ear. "
        "A distribution of t values was calculated by shuffling  condition "
        "labels %i times and for each test picking the largest absolute t-"
        "value across time and space. P-values were then calculated for "
        "every source and time point using this distribution." % n_samples)
section.append(text)
# image with significance maps in time bins
section = report.add_section("Result")
image = e.plot.brain.bin_table(pmap, tstep=0.05, surf='smoothwm', views=['lat', 'med'])
section.add_figure("Significant regions in time bins.", image)
# save the report
report.save_html("Source Permutation.html")
