'''
Created on Sep 30, 2012

@author: christian
'''
from copy import deepcopy
import fnmatch
import logging
import os
import shutil
import subprocess
import time

import numpy as np
from numpy import sin, cos
import scipy
from scipy.optimize import leastsq

from mayavi import mlab
from mayavi.tools import pipeline

import traits.api as traits
from traitsui.api import View, Item, HGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

import mne
from mne import fiff
from mne import write_trans
from mne.fiff import write
from mne.fiff.constants import FIFF

from eelbrain import load
from eelbrain import ui



class dev_head_viewer(traits.HasTraits):
    """
    Mayavi viewer for modifying the device-to-head coordinate coregistration.

    """
    # views
    frontal = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # visibility
    head_shape = traits.Bool(True)
    mri = traits.Bool(True)

    # fitting
    _refit = traits.Bool(False)
    _0 = traits.Bool(True)
    _1 = traits.Bool(True)
    _2 = traits.Bool(True)
    _3 = traits.Bool(True)
    _4 = traits.Bool(True)

    _save = traits.Button()
    scene = traits.Instance(MlabSceneModel, ())

    def __init__(self, raw, mrk, mri=None, trans=None):
        """
        Parameters
        ----------

        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.
        mrk : load.kit.marker_avg_file | str(path)
            marker_avg_file object, or path to a marker file.

        """
        traits.HasTraits.__init__(self)
        self.configure_traits()

        self.fitter = dev_head_fitter(raw, mrk, mri, trans)
        self.scene.disable_render = True
        self.fitter.plot(fig=self.scene.mayavi_scene)
        self._current_fit = None

        self.frontal = True
        self.scene.disable_render = False

    @traits.on_trait_change('head_shape')
    def _show_hs(self):
        self.fitter.headshape.set_opacity(int(self.head_shape))

    @traits.on_trait_change('mri')
    def _show_mri(self):
        if self.fitter.MRI:
            self.fitter.MRI.set_opacity(int(self.mri))
        else:
            ui.message("No MRI Loaded", "Load an MRI when initializing the "
                       "viewer", '!')

    @traits.on_trait_change('frontal')
    def _view_frontal(self):
        self.set_view('frontal')

    @traits.on_trait_change('left')
    def _view_left(self):
        self.set_view('left')

    @traits.on_trait_change('top')
    def _view_top(self):
        self.set_view('top')

    @traits.on_trait_change('_refit,_0,_1,_2,_3,_4')
    def _fit(self):
        if not self._refit:
            if self._current_fit is not None:
                self.fitter.reset()
                self._current_fit = None
            return

        idx = np.array([self._0, self._1, self._2, self._3, self._4], dtype=bool)
        if np.sum(idx) < 3:
            ui.message("Not Enough Points Selected", "Need at least 3 points.",
                       '!')
            return

        self.fitter.fit(idx)
        self._current_fit = idx

    @traits.on_trait_change('_save')
    def save(self):
        self.fitter.save()

    def set_view(self, view='frontal'):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = .15
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=450, width=450, show_label=False),
                HGroup('top', 'frontal', 'left',),
                HGroup('head_shape', 'mri'),
                HGroup('_refit', '_0', '_1', '_2', '_3', '_4'),
                HGroup('_save'),
                )



class mri_head_viewer(traits.HasTraits):
    """
    Mayavi viewer for fitting an MRI to a digitized head shape.

    """
    # views
    frontal = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # fitting
    nasion = traits.Array(float, (1, 3))
    rotation = traits.Array(float, (1, 3))
    scale = traits.Array(float, (1, 3), [[1, 1, 1]])

    fit_scale = traits.Button()
    fit_no_scale = traits.Button()

    _save = traits.Button()
    _save_trans = traits.Button()

    scene = traits.Instance(MlabSceneModel, ())

    def __init__(self, s_from, raw, s_to=None, subjects_dir=None):
        """
        Parameters
        ----------

        s_from : str
            name of the source subject (e.g., 'fsaverage')
        raw : str(path)
            path to a raw file containing the digitizer data.
        s_to : str | None
            Name of the the subject for which the MRI is destined (used to
            save MRI and in the trans file's file name). The defaut (`None`)
            is identical with s_from.
        subjects_dir : None | path
            Override the SUBJECTS_DIR environment variable
            (sys.environ['SUBJECTS_DIR'])

        """
        traits.HasTraits.__init__(self)
        self.configure_traits()

        self.fitter = mri_head_fitter(s_from, raw, s_to, subjects_dir)

        self.scene.disable_render = True
        self.fitter.plot(fig=self.scene.mayavi_scene)
        self.frontal = True
        self.scene.disable_render = False

    @traits.on_trait_change('fit_scale,fit_no_scale')
    def _fit(self, caller, info2):
        if caller == 'fit_scale':
            self.fitter.fit(method='mr')
        elif caller == 'fit_no_scale':
            self.fitter.fit(method='r')
        else:
            ui.message("Error", "Unknown caller for _fit(): %r" % caller, '!')

        self.rotation = [self.fitter.get_rot()]
        self.scale = [self.fitter.get_scale()]

    @traits.on_trait_change('_save')
    def save(self):
        self.fitter.save()

    @traits.on_trait_change('_save_trans')
    def save_trans(self):
        self.fitter.save_trans()

    @traits.on_trait_change('nasion')
    def set_nasion(self):
        args = tuple(self.nasion[0])
        self.fitter.set_nasion(*args)

    @traits.on_trait_change('scale,rotation')
    def set_trans(self):
        args = tuple(self.rotation[0]) + tuple(self.scale[0])
        self.fitter.set(*args)

    @traits.on_trait_change('top,left,frontal')
    def set_view(self, view='frontal', info=None):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = 150
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                HGroup('top', 'frontal', 'left',),
                HGroup('fit_scale', 'fit_no_scale'),
                HGroup('nasion'),
                HGroup('scale'),
                HGroup('rotation'),
                HGroup('_save', '_save_trans'),
                )



class coreg(traits.HasTraits):
    """

    http://docs.enthought.com/mayavi/mayavi/building_applications.html#making-the-visualization-live
    """
    # views
    frontal = traits.Button()
    left = traits.Button()
    top = traits.Button()

    # visibility
    fiducials = traits.Bool(True)
    dig_points = traits.Bool(True)
    head_shape = traits.Bool(True)
    scalp = traits.Bool(True)
    scalp_alpha = traits.Range(0., 1., 1.)
    sensors = traits.Bool(True)

    scene = traits.Instance(MlabSceneModel, ())

    def __init__(self, raw, fwd, bem=None):
        """
        Parameters
        ----------

        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.
        fwd : dict | str(path)
            MNE forward solution (returned by mne.read_forward_solution),
            or path to an MNE forward solution.
        bem : None | list | str(path)
            Bem file for the scalp surface: list as returned by
            mne.read_bem_surfaces, or path to a bem file.

        """
        traits.HasTraits.__init__(self)

        if isinstance(raw, basestring):
            raw = load.fiff.Raw(raw)
        if isinstance(fwd, basestring):
            fwd = mne.read_forward_solution(fwd)
        if isinstance(bem, basestring):
            bem = mne.read_bem_surfaces(bem)

        points3d = self.scene.mlab.points3d

        dev_2_head = fwd['info']['dev_head_t']['trans']
        head_2_dev = np.linalg.inv(dev_2_head)
        mri_2_head = fwd['mri_head_t']['trans']

        # sensors
        s = load.fiff.sensor_net(raw)
        x, y, z = s.locs.T
        self._sensors = points3d(x, y, z, scale_factor=0.005, color=(0, .2, 1))

        # head shape
        pts = filter(lambda d: d['kind'] == 4, raw.info['dig'])
        pts = np.array([d['r'] for d in raw.info['dig']])
        pts = np.hstack((pts, np.ones((len(pts), 1))))
        x, y, z, _ = np.dot(head_2_dev, pts.T)
        pts = points3d(x, y, z, opacity=0)  # color=(1,0,0), scale_factor=0.005)
        d = self.scene.mlab.pipeline.delaunay3d(pts)
        self._head_shape = self.scene.mlab.pipeline.surface(d)

        # scalp (mri-headshape)
        if bem:
            surf = bem[0]
            pts = surf['rr']
            pts = np.hstack((pts, np.ones((len(pts), 1))))
            pts = np.dot(mri_2_head, pts.T)
            pts = np.dot(head_2_dev, pts)
            x, y, z, _ = pts
            faces = surf['tris']
            self._bem = self.scene.mlab.triangular_mesh(x, y, z, faces, color=(.8, .8, .8), opacity=1)

        # fiducials
        pts = filter(lambda d: d['kind'] == 1, raw.info['dig'])
        pts = np.vstack([d['r'] for d in pts])
        pts = np.hstack((pts, np.ones((len(pts), 1))))
        pts = np.dot(head_2_dev, pts.T)
        x, y, z, _ = pts
        self._fiducials = points3d(x, y, z, color=(0, 1, 1), opacity=0.5)

        # dig points
        pts = filter(lambda d: d['kind'] == 2, raw.info['dig'])
        pts = np.vstack([d['r'] for d in pts])
        pts = np.hstack((pts, np.ones((len(pts), 1))))
        pts = np.dot(head_2_dev, pts.T)
        x, y, z, _ = pts
        self._dig_pts = points3d(x, y, z, color=(1, 0, 0), opacity=0.5)

        self.configure_traits()
        self.frontal = True
        if bem:
            self.head_shape = False

    @traits.on_trait_change('dig_points')
    def _show_dig_pts(self):
        self._dig_pts.actor.visible = self.dig_points

    @traits.on_trait_change('fiducials')
    def _show_fiducials(self):
        self._fiducials.actor.visible = self.fiducials

    @traits.on_trait_change('head_shape')
    def _show_hs(self):
        self._head_shape.actor.visible = self.head_shape

    @traits.on_trait_change('scalp')
    def _show_bem(self):
        if not hasattr(self, '_bem'):
            return
        self._bem.actor.visible = self.scalp

    @traits.on_trait_change('sensors')
    def _show_sensors(self):
        self._sensors.visible = self.sensors

    @traits.on_trait_change('scalp_alpha')
    def _set_scalp_alpha(self):
        self._bem.actor.property.opacity = self.scalp_alpha

    @traits.on_trait_change('frontal')
    def _view_frontal(self):
        self.set_view('frontal')

    @traits.on_trait_change('left')
    def _view_left(self):
        self.set_view('left')

    @traits.on_trait_change('top')
    def _view_top(self):
        self.set_view('top')

    def set_view(self, view='frontal'):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = .15
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180, reset_roll=True)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=450, width=450, show_label=False),
                HGroup('top', 'frontal', 'left',),
                HGroup('sensors', 'head_shape'),
                HGroup('scalp', 'scalp_alpha'),
                HGroup('fiducials', 'dig_points'),
                )



class mri_head_fitter:
    """
    Fit an MRI to a head shape model.

    Distances are internally represented in mm and converted where needed.

    Transforms applied to MRI:

    1) move MRI nasion to origin
    2) apply scaling
    3) apply rotation
    2) move MRI nasion to headshape nasion

    """
    def __init__(self, s_from, raw, s_to=None, subjects_dir=None):
        """
        Parameters
        ----------

        s_from : str
            name of the source subject (e.g., 'fsaverage')
        raw : str(path)
            path to a raw file containing the digitizer data.
        s_to : str | None
            Name of the the subject for which the MRI is destined (used to
            save MRI and in the trans file's file name). The defaut (`None`)
            is identical with s_from.
        subjects_dir : None | path
            Override the SUBJECTS_DIR environment variable
            (sys.environ['SUBJECTS_DIR'])

        """
        # interpret paths
        if subjects_dir is None:
            if 'SUBJECTS_DIR' in os.environ:
                subjects_dir = os.environ['SUBJECTS_DIR']
            else:
                err = ("If SUBJECTS_DIR is not set as environment variable, "
                       "it must be provided as subjects_dir parameter")
                raise ValueError(err)

        # MRI head shape
        fname = os.path.join(subjects_dir, s_from, 'bem', 'outer_skin.surf')
        pts, tri = mne.read_surface(fname)
        self.mri_hs = geom(pts, tri)

        fname = os.path.join(subjects_dir, s_from, 'bem', s_from + '-fiducials.fif')
        dig, _ = read_fiducials(fname)
        self.mri_fid = geom_fid(dig, unit='mm')

        # digitizer data from raw
        self._raw = raw
        raw = mne.fiff.Raw(raw)
        pts = filter(lambda d: d['kind'] == 4, raw.info['dig'])
        pts = np.array([d['r'] for d in pts]) * 1000
        self.dig_hs = geom(pts)
        self.dig_fid = geom_fid(raw.info['dig'], unit='mm')

        # move to the origin
        self.mri_o_t = trans(*self.mri_fid.nas).I
        self.o_dig_t = trans(*self.dig_fid.nas)

        self.subjects_dir = subjects_dir
        self.s_from = s_from
        self.s_to = s_to

        self.nas_t = trans(0, 0, 0)
        self.set(0, 0, 0, 1, 1, 1)

    def plot(self, size=(512, 512), fig=None):
        if fig is None:
            fig = mlab.figure(size=size)

        self.fig = fig
        self.mri_hs.plot_solid(fig)
        self.mri_fid.plot_points(fig, scale=5)
        self.dig_hs.plot_solid(fig, opacity=.5, rep='wireframe')
        self.dig_fid.plot_points(fig, scale=40, opacity=.25)
        return fig

    def _error(self, T):
        "For each point in pts, the distance to the closest point in pts0"
#        err = np.empty(pts.shape)
#        for i, pos in enumerate(pts):
#            dist3d = pts0 - pos[None, :]
#            dist = np.sqrt(np.sum(dist3d ** 2, 1))
#            idx = dist.argmin()
#            err[i] = dist3d[idx]
        pts = self.dig_hs.get_pts()
        pts0 = self.mri_hs.get_pts(self.o_dig_t * T * self.nas_t * self.mri_o_t)
        Y = scipy.spatial.distance.cdist(pts, pts0, 'euclidean')
        dist = Y.min(axis=1)
        return dist

    def _dist_fixnas_mr(self, param):
        rx, ry, rz, mx, my, mz = param
        T = rot(rx, ry, rz) * mult(mx, my, mz)
        err = self._error(T)
        logging.debug("Params = %s -> Error = %s" % (param, np.sum(err ** 2)))
        return err

    def _dist_fixnas_r(self, param):
        rx, ry, rz = param
        T = rot(rx, ry, rz)
        err = self._error(T)
        logging.debug("Params = %s -> Error = %s" % (param, np.sum(err ** 2)))
        return err

    def _dist_fixnas_1mult(self, param):
        rx, ry, rz, m = param
        T = rot(rx, ry, rz) * mult(m, m, m)
        err = self._error(T)
        logging.debug("Params = %s -> Error = %s" % (param, np.sum(err ** 2)))
        return err

    def _estimate_fixnas_mr(self, params=(0, 0, 0, 1, 1, 1), **kwargs):
        params = np.asarray(params, dtype=float)
        est_params, self.info = leastsq(self._dist_fixnas_mr, params, **kwargs)
        return est_params

    def _estimate_fixnas_r(self, params=(0, 0, 0), **kwargs):
        params = np.asarray(params, dtype=float)
        est_params, self.info = leastsq(self._dist_fixnas_r, params, **kwargs)
        return est_params

    def _estimate_fixnas_1mult(self, params=(0, 0, 0, 1), **kwargs):
        params = np.asarray(params, dtype=float)
        est_params, self.info = leastsq(self._dist_fixnas_1mult, params, **kwargs)
        return est_params

    def fit(self, epsfcn=0.01, method='mr', **kwargs):
        """
        method : 'mr' | 'r'
            m: multiplication;
            r: rotationl;

        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        """
        t0 = time.time()
        if method == '1mult':
            est = self._estimate_fixnas_1mult(epsfcn=epsfcn, **kwargs)
            est = np.hstack((est, np.ones(2) * est[-1:]))
        elif method == 'mr':
            est = self._estimate_fixnas_mr(epsfcn=epsfcn, **kwargs)
        elif method == 'r':
            est = self._estimate_fixnas_r(epsfcn=epsfcn, **kwargs)
            est = tuple(est) + (1, 1, 1)
        else:
            raise ValueError("method")
        self.set(*est)
        dt = time.time() - t0
        print "%r took %.2f minutes" % (method, dt / 60)
        return est

    def get_T_trans(self, unit='mm'):
        "T for the trans file;, rot + translation "
        T0 = self.nas_t * self.mri_o_t
        if unit == 'mm':
            T = self.o_dig_t * self.trans_rot * T0
        elif unit == 'm':
            T0[:3, 3] /= 1000
            trans1 = self.o_dig_t.copy()
            trans1[:3, 3] /= 1000
            T = trans1 * self.trans_rot * T0
        else:
            raise ValueError('Unknown unit %r' % unit)
        return T.I

    def get_rot(self):
        return self._params[:3]

    def get_scale(self):
        return self._params[3:]

    def save(self, s_to=None, fwd_args=["--ico", '4', '--surf'], make_fwd=True):
        """
        s_to : None | str
            Override s_to set on initialization.

        fwd_args : list
            List of arguments for the `mne_setup_forward_model` call.

        make_fwd : bool | 'block'
            Call `mne_setup_forward_model` at the end. With True, the command
            is called and the corresponding Popen object returned. With
            'block', the Python interpreter is blocked until
            `mne_setup_forward_model` finishes.

        """
        s_from = self.s_from
        if s_to is None:
            if self.s_to is None:
                raise IOError("No destination specified")
            else:
                s_to = self.s_to

        # make sure we have an empty target directory
        sdir = os.path.join(self.subjects_dir, '{sub}')
        sdir_dest = sdir.format(sub=s_to)
        if os.path.exists(sdir_dest):
            msg = ("Subject directory exists: %r." % sdir_dest)
            if ui.ask("Overwrite MRI?", msg):
                shutil.rmtree(sdir_dest)
            else:
                raise IOError(msg)

        # find target paths
        bemdir = os.path.join(sdir, 'bem')
        os.makedirs(bemdir.format(sub=s_to))
        bempath = os.path.join(bemdir, '{name}.{ext}')
        surfdir = os.path.join(sdir, 'surf')
        os.mkdir(surfdir.format(sub=s_to))
        surfpath = os.path.join(surfdir, '{name}')

        # write parameters as text
        fname = os.path.join(sdir, 'T.txt').format(sub=s_to)
        with open(fname, 'w') as fid:
            fid.write(', '.join(map(str, self._params)))

        # write trans file
        self.save_trans(s_to=s_to)

        # MRI Scaling
        T0 = self.nas_t * self.mri_o_t
        T = T0.I * self.trans_scale * T0

        # assemble list of surface files to duplicate
        # surf/ files
        surf_names = ('orig', 'orig_avg',
                      'inflated', 'inflated_avg', 'inflated_pre',
                      'pial', 'pial_avg',
                      'smoothwm',
                      'white', 'white_avg',
                      'sphere', 'sphere.reg', 'sphere.reg.avg')
        paths = {}
        for name in surf_names:
            for hemi in ('lh.', 'rh.'):
                k = surfpath.format(sub=self.s_from, name=hemi + name)
                v = surfpath.format(sub=s_to, name=hemi + name)
                paths[k] = v

        # watershed files
        for surf in ['inner_skull', 'outer_skull', 'outer_skin']:
            k = bempath.format(sub=s_from, name=surf, ext='surf')
            k = os.path.realpath(k)
            v = bempath.format(sub=s_to, name=surf, ext='surf')
            paths[k] = v

        # make surf files [in mm]
        for src, dest in paths.iteritems():
            pts, tri = mne.read_surface(src)
            pts = apply_T(pts, T)
            mne.write_surface(dest, pts, tri)


        # write bem [in m]
        path = os.path.join(self.subjects_dir, '{sub}', 'bem', '{sub}-{name}.fif')
        for name in ['head']:  # '5120-bem-sol',
            src = path.format(sub=s_from, name=name)
            dest = path.format(sub=s_to, name=name)
            surf = mne.read_bem_surfaces(src)[0]
            surf['rr'] = apply_T(surf['rr'], T)
            mne.write_bem_surface(dest, surf)

        # fiducials [in m]
        fname = path.format(sub=s_from, name='fiducials')
        pts, cframe = read_fiducials(fname)
        for pt in pts:
            pt['r'] = apply_T_1pt(pt['r'], T, scale=.001)
        fname = path.format(sub=s_to, name='fiducials')
        write_fiducials(fname, pts, cframe)

        # write src
        path = os.path.join(self.subjects_dir, '{sub}', 'bem', '{sub}-ico-4-src.fif')
        src = path.format(sub=s_from)
        sss = mne.read_source_spaces(src)
        for ss in sss:
            ss['rr'] = apply_T(ss['rr'], T)
            ss['nn'] = apply_T(ss['nn'], T.I.T)
        dest = path.format(sub=s_to)
        mne.write_source_spaces(dest, sss)

        # Labels [in m]
        lbl_dir = os.path.join(self.subjects_dir, '{sub}', 'label')
        top = lbl_dir.format(sub=s_from)
        relpath_start = len(top) + 1
        dest_top = lbl_dir.format(sub=s_to)
        lbls = []
        for dirp, _, files in os.walk(top):
            files = fnmatch.filter(files, '*.label')
            dirp = dirp[relpath_start:]
            lbls.extend(map(os.path.join(dirp, '{0}').format, files))
        for lbl in lbls:
            l = mne.read_label(os.path.join(top, lbl))
            pos = apply_T(l.pos, T, scale=.001)
            l2 = mne.Label(l.vertices, pos, l.values, l.hemi, l.comment)
            dest = os.path.join(dest_top, lbl)
            dirname, _ = os.path.split(dest)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            l2.save(dest)

        # duplicate curvature files
        path = os.path.join(self.subjects_dir, '{sub}', 'surf', '{name}')
        for name in ['lh.curv', 'rh.curv']:
            src = path.format(sub=s_from, name=name)
            dest = path.format(sub=s_to, name=name)
            shutil.copyfile(src, dest)

        # run mne_setup_forward_model
        if not make_fwd:
            return

        cmd = ["mne_setup_forward_model", "--subject", "R0040"] + fwd_args

        if make_fwd == 'block':
            subprocess.call(cmd)
        else:
            return subprocess.Popen(cmd)

    def save_trans(self, fname=None, s_to=None):
        """
        Save only the trans file (with structural MRI)
        """
        if fname is None:
            if s_to is None:
                if self.s_to is None:
                    raise IOError("No destination specified")
                else:
                    s_to = self.s_to
            rawdir = os.path.dirname(self._raw)
            fname = os.path.join(rawdir, s_to + '-trans.fif')

        if os.path.exists(fname):
            msg = ("Trans file exists: %r" % fname)
            if ui.ask("Overwrite Trans?", msg):
                os.remove(fname)
            else:
                raise IOError(msg)

        # in m
        trans = self.get_T_trans('m')
        dig = deepcopy(self.dig_fid.source_dig)  # these are in m
        for d in dig:
            coord = apply_T_1pt(d['r'], trans, scale=1. / 1000)
            d['r'] = coord[:3, 0]
        info = {'to':FIFF.FIFFV_COORD_MRI, 'from': FIFF.FIFFV_COORD_HEAD,
                'trans': np.array(trans), 'dig': dig}
        write_trans(fname, info)

    def set(self, rx, ry, rz, mx, my, mz):
        self._params = (rx, ry, rz, mx, my, mz)
        self.trans_rot = rot(rx, ry, rz)
        self.trans_scale = mult(mx, my, mz)
        self.update()

    def set_nasion(self, x, y, z):
        self.nas_t = trans(x, y, z)
        self.update()

    def update(self):
        T = self.o_dig_t * self.trans_rot * self.trans_scale * self.nas_t * self.mri_o_t
        T = T.I
        for g in [self.dig_hs, self.dig_fid]:
            g.set_T(T)



def apply_T_1pt(X, T, scale=1):
    X = np.vstack((X[:, None], [1]))
    if scale != 1:
        X[:3] *= scale

    X = T * X
    X = X[:3, 0]

    X = np.array(X)
    if scale != 1:
        X /= scale
    return X


def apply_T(X, T, scale=1):
    X = np.vstack((X.T, np.ones(len(X))))
    if scale != 1:
        X[:3] *= scale

    X = T * X
    X = X[:3].T

    X = np.array(X)
    if scale != 1:
        X /= scale
    return X


def trans(x=0, y=0, z=0):
    "MNE manual p. 95"
    m = np.matrix([[1, 0, 0, x],
                   [0, 1, 0, y],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])
    return m

def rot(x=0, y=0, z=0):
    r = np.matrix([[cos(y) * cos(z), -cos(x) * sin(z) + sin(x) * sin(y) * cos(z), sin(x) * sin(z) + cos(x) * sin(y) * cos(z), 0],
                  [cos(y) * sin(z), cos(x) * cos(z) + sin(x) * sin(y) * sin(z), -sin(x) * cos(z) + cos(x) * sin(y) * sin(z), 0],
                  [-sin(y), sin(x) * cos(y), cos(x) * cos(y), 0],
                  [0, 0, 0, 1]])
    return r

def mult(x=1, y=1, z=1):
    s = np.matrix([[x, 0, 0, 0],
                   [0, y, 0, 0],
                   [0, 0, z, 0],
                   [0, 0, 0, 1]])
    return s



def read_fiducials(fname):
    """
    Read fiducials from a fiff file


    Returns
    -------
    pts : list of dicts
        List of digitizer points (each point in a dict).
    coord_frame : int
        The coordinate frame of the points (see below).


    MNE Coordinate Frames
    ---------------------

    1  FIFFV_COORD_DEVICE
    2  FIFFV_COORD_ISOTRAK
    3  FIFFV_COORD_HPI
    4  FIFFV_COORD_HEAD
    5  FIFFV_COORD_MRI
    6  FIFFV_COORD_MRI_SLICE
    7  FIFFV_COORD_MRI_DISPLAY
    8  FIFFV_COORD_DICOM_DEVICE
    9  FIFFV_COORD_IMAGING_DEVICE
    0  FIFFV_COORD_UNKNOWN

    """
    fid, tree, _ = fiff.open.fiff_open(fname)
    isotrak = fiff.tree.dir_tree_find(tree, FIFF.FIFFB_ISOTRAK)
    isotrak = isotrak[0]
    pts = []
    coord_frame = 0
    for k in range(isotrak['nent']):
        kind = isotrak['directory'][k].kind
        pos = isotrak['directory'][k].pos
        if kind == FIFF.FIFF_DIG_POINT:
            tag = fiff.tag.read_tag(fid, pos)
            pts.append(tag.data)
        elif kind == FIFF.FIFF_MNE_COORD_FRAME:
            tag = fiff.tag.read_tag(fid, pos)
            coord_frame = tag.data[0]

    fid.close()
    return pts, coord_frame

def write_fiducials(fname, dig, coord_frame=0):
    """
    Write
    """
    fid = write.start_file(fname)
    write.start_block(fid, FIFF.FIFFB_ISOTRAK)
    write.write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, coord_frame)
    for pt in dig:
        write.write_dig_point(fid, pt)

    write.end_block(fid, FIFF.FIFFB_ISOTRAK)
    write.end_file(fid)



class geom(object):
    """
    Represents a set of points and a list of transformations, and can plot the
    points as points or as surface to a mayavi figure.

    """
    def __init__(self, pts, tri=None):
        """
        pts : array, shape = (n_pts, 3)
            A list of points
        tri : None | array, shape = (n_tri, 3)
            Triangularization (optional). A list of triangles, each triangle
            composed of the indices of three points forming a triangle
            together.

        """
        self.trans = []

        self.pts = pts = np.vstack((pts.T, np.ones(len(pts))))
        self.tri = tri

        self._plots_surf = []
        self._plots_pt = []

    def get_pts(self, T=None):
        """
        returns the points contained in the object


        Parameters
        ----------

        T : None | true | Matrix (4x4)
            None: don't transform the points
            True: apply the transformation matrix that is stored in the object
            matrix: apply the given transformation matrix


        Returns
        -------

        pts : array, shape = (n_pts, 3)
            The points.

        """
        if T is True:
            T = self.get_T()

        if T is None:
            pts = self.pts
        else:
            pts = T * self.pts

        return np.array(pts[:3].T)

    def get_T(self):
        "complete transformation"
        T = np.matrix(np.eye(4))
        for Ti in self.trans:
            T = Ti * T
        return T

    def plot_solid(self, fig, opacity=1., rep='surface', color=(1, 1, 1)):
        "Returns: mesh, surf"
        if self.tri is None:
            d = scipy.spatial.Delaunay(self.pts[:3].T)
            self.tri = d.convex_hull

        x, y, z, _ = self.pts

        mesh = pipeline.triangular_mesh_source(x, y, z, self.tri, figure=fig)
        surf = pipeline.surface(mesh, figure=fig, color=color, opacity=opacity,
                                representation=rep)

        self._plots_surf.append((mesh, surf))
        if self.trans:
            self.update_plot()

        return mesh, surf

    def plot_points(self, fig, scale=1e-2, opacity=1., color=(1, 0, 0)):
        "Returns: src, glyph"
        x, y, z, _ = self.pts

        src = pipeline.scalar_scatter(x, y, z)
        glyph = pipeline.glyph(src, color=color, figure=fig, scale_factor=scale,
                               opacity=opacity)

        self._plots_pt.append((src, glyph))
        if self.trans:
            self.update_plot()

        return src, glyph

    def add_T(self, T):
        self.trans.append(T)
        self.update_plot()

    def reset_T(self):
        self.trans = []
        self.update_plot()

    def set_opacity(self, v=1):
        if v == 1:
            v = True
        elif v == 0:
            v = False
        else:
            raise NotImplementedError

        for _, plt in self._plots_pt + self._plots_surf:
            if isinstance(v, bool):
                plt.visible = v

    def set_T(self, T):
        """
        T : list | transformation matrix, shape = (4,4)
            The transformation to be applied, or a list of transformations.

        """
        if not isinstance(T, (list)):
            T = [T]

        for Ti in T:
            assert Ti.shape == (4, 4)

        self.trans = T
        self.update_plot()

    def update_plot(self):
        pts = self.get_pts(T=True)
        for mesh, _ in self._plots_surf:
            mesh.data.points = pts
        for src, _ in self._plots_pt:
            src.data.points = pts



class geom_fid(geom):
    def __init__(self, dig, unit='mm'):
        if unit == 'mm':
            x = 1000
        elif unit == 'm':
            x = 1
        else:
            raise ValueError('Unit: %r' % unit)

        dig = filter(lambda d: d['kind'] == 1, dig)
        pts = np.array([d['r'] for d in dig]) * x

        super(geom_fid, self).__init__(pts)
        self.unit = unit
        self.dig = dig

        self.source_dig = dig
        digs = {d['ident']: d for d in dig}
        self.rap = digs[1]['r'] * x
        self.nas = digs[2]['r'] * x
        self.lap = digs[3]['r'] * x



class dev_head_fitter:
    def __init__(self, raw, mrk, mri=None, trans=None):
        """
        Parameters
        ----------

        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.
        mrk : load.kit.marker_avg_file | str(path)
            marker_avg_file object, or path to a marker file.
        mri : None |
            Only for visualization purposes.


        """
        # interpret mrk
        if isinstance(mrk, basestring):
            mrk = load.kit.marker_avg_file(mrk)

        # interpret raw
        if isinstance(raw, basestring):
            self._raw_fname = raw
            raw = load.fiff.Raw(raw)
        else:
            self._raw_fname = raw.info['filename']
        self.raw = raw

        # mri-head-trans
        if mri is None:
            self.MRI = None
        else:
            if not isinstance(trans, dict):
                trans = mne.read_trans(trans)
            T = np.matrix(trans['trans'])
            self.T_mri_head = T.I

            # interpret mri
            if isinstance(mri, basestring):
                s = mne.read_bem_surfaces(mri)[0]
                pts, tri = s['rr'], s['tris']
            else:
                pts, tri = mri
            self.MRI = geom(pts, tri=tri)
            self.MRI.set_T(self.T_mri_head)

        # sensors
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_MEG_CH, raw.info['chs'])
        pts = np.array([d['loc'][:3] for d in pts])
        self.sensors = geom(pts)

        # marker points
        pts = mrk.points / 1000
        pts = pts[:, [1, 0, 2]]
        pts[:, 0] *= -1
        self.mrk = geom(pts)

        # head shape
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_POINT_EXTRA, raw.info['dig'])
        pts = np.array([d['r'] for d in pts])
        self.headshape = geom(pts)

        # HPI points
        pts = filter(lambda d: d['kind'] == FIFF.FIFFV_POINT_HPI, raw.info['dig'])
        assert [d['ident'] for d in pts] == range(1, 6)
        pts = np.array([d['r'] for d in pts])
        self.HPI = geom(pts)

        # T head-to-device
        trans = raw.info['dev_head_t']['trans']
        self.T_head2dev = np.matrix(trans).I
        self.reset()
        self._HPI_flipped = False

    def fit(self, include=range(5)):
        """
        Fit the marker points to the digitizer points.

        include : index (numpy compatible)
            Which points to include in the fit. Index should select among
            points [0, 1, 2, 3, 4].
        """
        def err(params):
            T = trans(*params[:3]) * rot(*params[3:])
            est = self.HPI.get_pts(T)[include]
            tgt = self.mrk.get_pts()[include]
            return (tgt - est).ravel()

        # initial guess
        params = (0, 0, 0, 0, 0, 0)
        params, _ = leastsq(err, params)
        self.est_params = params

        T = trans(*params[:3]) * rot(*params[3:])
        self.est_T = T

        self.headshape.set_T(T)
        self.HPI.set_T(T)
        if self.MRI:
            self.MRI.set_T(T * self.T_mri_head)

    def plot(self, size=(800, 800), fig=None, HPI_ns=False):
        """
        Plot sensor helmet and head. ``fig`` is used if provided, otherwise
        a new mayavi figure is created with ``size``.

        HPI_ns : bool
            Add number labels to the HPI points.

        """
        if fig is None:
            fig = mlab.figure(size=size)

        self.mrk.plot_points(fig, scale=1.1e-2, opacity=.5, color=(1, 0, 0))
        self.sensors.plot_points(fig, scale=1e-2, color=(0, 0, 1))

        self.HPI.plot_points(fig, scale=1e-2, color=(1, .8, 0))
        self.headshape.plot_solid(fig, opacity=1., color=(1, 1, 1))

        if self.MRI is not None:
            self.MRI.plot_solid(fig, opacity=1., color=(.6, .6, .5))

        # label marker points
        for i, pt in enumerate(self.mrk.pts[:3].T):
            x, y, z = pt
            self.txt = mlab.text3d(x, y, z, str(i), scale=.01)

        if HPI_ns:  # label HPI points
            for i, pt in enumerate(self.HPI.pts[:3].T):
                x, y, z = pt
                mlab.text3d(x, y, z, str(i), scale=.01, color=(1, .8, 0))

        return fig

    def reset(self):
        """
        Reset the current device-to-head transformation to the one contained
        in the raw file

        """
        T = self.T_head2dev
        self.headshape.set_T(T)
        self.HPI.set_T(T)
        if self.MRI:
            self.MRI.set_T(T * self.T_mri_head)

    def save(self, fname=None):
        """
        Save a copy of the raw file with the current device-to-head
        transformation

        """
        if fname is None:
            msg = "Destination for the modified raw file"
            ext = [('fif', 'MNE Fiff File')]
            fname = ui.ask_saveas("Save Raw File", msg, ext,
                                  default=self._raw_fname)
        if not fname:
            return

        info = self.raw.info
        info['dev_head_t']['trans'] = np.array(self.est_T.I)

        self.raw.save(fname)

    def set_hs_opacity(self, v=1):
        self.headshape.set_opacity(v)



class set_nasion(traits.HasTraits):
    """
    Mayavi viewer for modifying the device-to-head coordinate coregistration.

    """
    # views
    frontal = traits.Button()
    left = traits.Button()
    top = traits.Button()

#    coord = traits.Array(float, (1, 3))
    x = traits.Float()
    y = traits.Float()
    z = traits.Float()

    _save = traits.Button()
    scene = traits.Instance(MlabSceneModel, ())

    def __init__(self, subject, subjects_dir=None):
        """
        Parameters
        ----------

        raw : mne.fiff.Raw | str(path)
            MNE Raw object, or path to a raw file.
        mrk : load.kit.marker_avg_file | str(path)
            marker_avg_file object, or path to a marker file.

        """
        traits.HasTraits.__init__(self)
        self.configure_traits()

        self.scene.disable_render = True

        if subjects_dir is None:
            subjects_dir = os.environ['SUBJECTS_DIR']
        self.subjects_dir = subjects_dir
        self.subject = subject

        fname = os.path.join(subjects_dir, subject, 'bem', subject + '-head.fif')
        s = mne.read_bem_surfaces(fname)[0]
        self._pts = s['rr']
        self.head = geom(s['rr'], tri=s['tris'])
        self.head_mesh, _ = self.head.plot_solid(self.scene.mayavi_scene,
                                                 color=(.7, .7, .6))

        self.nasion = pipeline.scalar_scatter(0, 0, 0)
        glyph = pipeline.glyph(self.nasion, figure=self.scene.mayavi_scene,
                               color=(1, 0, 0), opacity=0.8, scale_factor=0.01)

#        self.nasion.data.points.sync_trait('data', self.coord)

        picker = self.scene.mayavi_scene.on_mouse_pick(self._on_mouse_click)
        self._current_fit = None

        self.frontal = True
        self.scene.disable_render = False

    def _on_mouse_click(self, picker):
        l = dir(picker)
        l = filter(lambda x: not x.startswith('_'), l)
        picked = picker.actors
        pid = picker.point_id
        self.x, self.y, self.z = self._pts[pid]

    @traits.on_trait_change('x,y,z')
    def on_nasion_change(self):
        self.nasion.data.points = [(self.x, self.y, self.z)]

    @traits.on_trait_change('frontal')
    def _view_frontal(self):
        self.set_view('frontal')

    @traits.on_trait_change('left')
    def _view_left(self):
        self.set_view('left')

    @traits.on_trait_change('top')
    def _view_top(self):
        self.set_view('top')

    @traits.on_trait_change('_save')
    def _on_dave(self):
        self.save()

    def save(self, fname=None, overwrite=False):
        if fname is None:
            fname = os.path.join(self.subjects_dir, self.subject, 'bem',
                                 self.subject + '-fiducials.fif')

        if os.path.exists(fname) and not overwrite:
            if not ui.ask("Overwrite?", "File already exists: %r" % fname):
                return


        dig = [
               {'kind': 1, 'ident': 1, 'r': np.array([.08, 0, 0])},
               {'kind': 1, 'ident': 2, 'r': np.array([self.x, self.y, self.z])},
               {'kind': 1, 'ident': 3, 'r': np.array([-.08, 0, 0])},
               ]
        write_fiducials(fname, dig, FIFF.FIFFV_COORD_MRI)

    def set_view(self, view='frontal'):
        self.scene.parallel_projection = True
        self.scene.camera.parallel_scale = .15
        kwargs = dict(azimuth=90, elevation=90, distance=None, roll=180,
                      reset_roll=True, figure=self.scene.mayavi_scene)
        if view == 'left':
            kwargs.update(azimuth=180, roll=90)
        elif view == 'top':
            kwargs.update(elevation=0)
        self.scene.mlab.view(**kwargs)

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=600, show_label=False),
                HGroup('top', 'frontal', 'left',),
                HGroup('x', 'y', 'z',),
#                HGroup('coord',),
                HGroup('_save'),
                )
