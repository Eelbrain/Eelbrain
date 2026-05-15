# requires: mayavi
# dataset: mne_sample
'''
Example performes a permutation cluster test on source space data and creates
an HTML file describing he output.
'''
import eelbrain as e


# settings
n_samples = 1000


# create an HTML report in which to document results
report = e.Report("MNE Sample Dataset", author="Prof. Enid Gumby")
section = report.add_section("Introduction")
text = ("A comparison of auditory stimulation to the left vs. the right ear "
        "in the MNE sample dataset. "
        "Spatio-temporal clusters were formed by thresholding the comparison "
        "at a t-value equivalent to an uncorrected p-value of 0.05. For each "
        "cluster, a cluster value was calculated as the sum of all t-values "
        "in the cluster. In order to calculate p-values for each cluster, "
        "a distribution of cluster values was computed by shuffling condition "
        "labels %i times and extracting each time the value of the largest "
        "cluster." % n_samples)
section.append(text)

'''
use the sample dataset loader to load source space data for the mne samples
dataset. Load only auditory data.
'''
ds = e.datasets.get_mne_sample(-0.1, 0.2, src='ico', sub="modality == 'A'")

# Add a table with trial numbers to the report
freqs = e.table.frequencies('side', data=ds)
freq_table = freqs.as_table()
section.add_figure("Number of trials in each condition", freq_table)


'''
perform t-test for side of stimulation (for a more reliable test set samples=10000).
'''
res = e.testnd.TTestIndependent(
    'src', 'side', 'L', 'R', data=ds,
    samples=n_samples,  # number of permutations
    pmin=0.05,  # threshold for clusters (uncorrected p-value)
    tstart=0.05,  # start of the time window of interest
    mintime=0.02,  # minimum duration for clusters
    minsource=10,  # minimum number of sources for clusters
)

# add results to the report
section = report.add_section("Result")

# add an image with all clusters in time bins
pmap = res.masked_parameter_map(1)
image = e.plot.brain.bin_table(pmap, tstep=0.05, surf='smoothwm', views=['lat', 'med'])
section.add_figure("Significant clusters in time bins.", image)
# add a table of all clusters
cluster_table = res.clusters.as_table()
section.add_figure("All clusters", cluster_table)

# plot clusters with p values smaller than 0.25 separately
clusters = res.clusters.sub("p < 0.25")
for i in range(clusters.n_cases):
    p = clusters[i, 'p']

    # add a section with appropriate title
    title = "Cluster %i, p=%.2f" % (i, p)
    subsection = section.add_section(title)

    # retrieve the cluster
    c_i = clusters[i, 'cluster']
    # plot the extent
    c_extent = c_i.sum('time')
    brain = e.plot.brain.cluster(c_extent)
    # add to report (plot.brain plots can be captured using Brain.image())
    image = brain.image("cluster 0 extent")
    brain.close()
    subsection.add_figure(f"Extent the cluster, p={p}", image)

    # extract and analyze the value in the cluster in each trial
    index = c_i != 0
    c_value = ds['src'].sum(index)
    # index is a boolean NDVar over space and time, so here we are summing in the
    # whole spatio-temporal cluster
    plt_box = e.plot.Boxplot(c_value, 'side', data=ds)
    pw_table = e.test.pairwise(c_value, 'side', data=ds)
    # add to report (create an image from an eelbrain matplotlib plot with .image())
    # use PNG because SVG boxplots do not render correctly in Safari
    image = plt_box.image('image.png')
    section.add_figure("Cluster value", [image, pw_table])

    # extract and analyze the time course in the cluster
    index = c_extent != 0
    c_timecourse = ds['src'].sum(index)
    # c_extent is a boolean NDVar over space only, so here we are summing over the
    # spatial extent of the cluster for every time point but keep the time dimension
    plt_tc = e.plot.UTSStat(c_timecourse, 'side', data=ds)
    # add to report
    image = plt_tc.image()
    section.add_figure("Time course of the average in the largest cluster "
                       "extent", image)

# save the report
report.save_html("Source Permutation Clusters.html")
