# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# This script can be executed from the Terminal using $ python
# dataset: mne_sample
from eelbrain import *


w = 3

# load sample data
ds = datasets.get_mne_sample(sub="modality=='A'", sns=True)


# interactive plot
plot.TopoButterfly('meg', data=ds)


# plot topography at a specific time point
plot.Topomap("meg.sub(time=.07)", data=ds)

# plot topography at a specific time point by condition
plot.Topomap("meg.sub(time=.07)", 'side', data=ds)

# plot topography at a specific time point for one condition;
# clip with a circular outline
plot.Topomap("meg.sub(side=='L', time=.07)", data=ds, clip='circle')

# plot topography with a specific projection
plot.Topomap("meg.sub(time=.07)", proj='back', data=ds)

# same but without clipping the map
plot.Topomap("meg.sub(time=.07)", proj='back', clip=False, data=ds)


# mark sensors (all sensors with z coordinate < 0
y = ds['meg'].sub(time=.07).mean('case')
mark = ds['meg'].sensor.z < 0
plot.Topomap(y, mark=mark)

# different cmap, and manually adjusted head outline
plot.Topomap(y, head_radius=(0.3, 0.33), head_pos=0.02, cmap='jet')

# sensor labels
plot.SensorMap(y, mark=mark, head_radius=(0.3, 0.33), head_pos=0.02,
               labels='name')

# sensor map with labels
p = plot.SensorMap(y, mark=mark, w=9, head_radius=(0.3, 0.33), head_pos=0.02)
# move overlapping labels apart
p.separate_labels()


# run the GUI if the script is executed from the shell
if __name__ == '__main__':
    gui.run()
