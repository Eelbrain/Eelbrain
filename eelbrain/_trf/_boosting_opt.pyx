# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
from libc.math cimport fabs
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
import numpy as np

cimport numpy as np

ctypedef np.int8_t INT8
ctypedef np.int64_t INT64
ctypedef np.float64_t FLOAT64

# Compiler and compiler flags used to build this extension (flags set in setup.py); they affect floating point results
cdef extern from *:
    """
    #ifndef EELBRAIN_BOOSTING_FLAGS
    #define EELBRAIN_BOOSTING_FLAGS ""
    #endif
    #if defined(__VERSION__)
    #define EELBRAIN_BOOSTING_COMPILER __VERSION__
    #elif defined(_MSC_FULL_VER)
    #define EELBRAIN_STRINGIFY_(x) #x
    #define EELBRAIN_STRINGIFY(x) EELBRAIN_STRINGIFY_(x)
    #define EELBRAIN_BOOSTING_COMPILER "MSVC " EELBRAIN_STRINGIFY(_MSC_FULL_VER)
    #else
    #define EELBRAIN_BOOSTING_COMPILER ""
    #endif
    """
    const char* EELBRAIN_BOOSTING_FLAGS
    const char* EELBRAIN_BOOSTING_COMPILER

COMPILE_FLAGS: tuple[str, ...] = tuple(EELBRAIN_BOOSTING_FLAGS.decode('ascii').split())
COMPILER: str = EELBRAIN_BOOSTING_COMPILER.decode('ascii')


cdef double inf = float('inf')


cdef double square(double x) noexcept nogil:
    # cf. https://github.com/scikit-image/scikit-image/issues/3026
    return x * x


cdef void l1_for_delta(
        FLOAT64 * y_error,
        FLOAT64 * x,
        double x_pad,  # pad x outside valid convolution area
        INT64 [:,:] indexes,  # training segment indexes
        double delta,
        int shift,  # TRF element offset
        double* e_add,
        double* e_sub,
    ) noexcept nogil:
    cdef:
        double d
        Py_ssize_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    e_add[0] = 0.
    e_sub[0] = 0.

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        if seg_start == -1:
            break
        seg_stop = indexes[seg_i, 1]
        # determine valid convolution segment
        conv_start = seg_start
        conv_stop = seg_stop
        if shift > 0:
            conv_start += shift
        elif shift < 0:
            conv_stop += shift
        # padding
        d = delta * x_pad
        # pre-
        for i in range(seg_start, conv_start):
            e_add[0] += fabs(y_error[i] - d)
            e_sub[0] += fabs(y_error[i] + d)
        # post-
        for i in range(conv_stop, seg_stop):
            e_add[0] += fabs(y_error[i] - d)
            e_sub[0] += fabs(y_error[i] + d)
        # valid segment
        for i in range(conv_start, conv_stop):
            d = delta * x[i - shift]
            e_add[0] += fabs(y_error[i] - d)
            e_sub[0] += fabs(y_error[i] + d)


cdef void l2_for_delta(
        FLOAT64 * y_error,
        FLOAT64 * x,
        double x_pad,  # pad x outside valid convolution area
        INT64 [:,:] indexes,  # training segment indexes
        double delta,
        int shift,
        double* e_add,
        double* e_sub,
    ) noexcept nogil:
    cdef:
        double d
        Py_ssize_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    e_add[0] = 0.
    e_sub[0] = 0.

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        if seg_start == -1:
            break
        seg_stop = indexes[seg_i, 1]
        # determine valid convolution segment
        conv_start = seg_start
        conv_stop = seg_stop
        if shift > 0:
            conv_start += shift
        elif shift < 0:
            conv_stop += shift
        # padding
        d = delta * x_pad
        # pre-
        for i in range(seg_start, conv_start):
            e_add[0] += square(y_error[i] - d)
            e_sub[0] += square(y_error[i] + d)
        # post-
        for i in range(conv_stop, seg_stop):
            e_add[0] += square(y_error[i] - d)
            e_sub[0] += square(y_error[i] + d)
        # part of the segment that is affected
        for i in range(conv_start, conv_stop):
            d = delta * x[i - shift]
            e_add[0] += square(y_error[i] - d)
            e_sub[0] += square(y_error[i] + d)


def error_for_indexes(
        FLOAT64 [:] x,
        INT64[:,:] indexes,  # (n_segments, 2)
        int error,  # 1 --> l1; 2 --> l2
):
    return error_for_indexes_c(&x[0], indexes, error)


cdef double error_for_indexes_c(
        FLOAT64 * x,
        INT64[:,:] indexes,  # (n_segments, 2)
        int error,  # 1 --> l1; 2 --> l2
) noexcept nogil:
    cdef:
        Py_ssize_t seg_i, i
        double out = 0

    if error == 1:
        for seg_i in range(indexes.shape[0]):
            if indexes[seg_i, 0] == -1:
                break
            for i in range(indexes[seg_i, 0], indexes[seg_i, 1]):
                out += fabs(x[i])
    else:
        for seg_i in range(indexes.shape[0]):
            if indexes[seg_i, 0] == -1:
                break
            for i in range(indexes[seg_i, 0], indexes[seg_i, 1]):
                out += x[i] * x[i]
    return out


def boosting_runs(
        FLOAT64 [:,:] y,  # (n_y, n_times)
        FLOAT64 [:,::1] x,  # (n_x, n_times)
        FLOAT64 [:] x_pads,  # (n_x,)
        INT64 [:,:,:] split_train,
        INT64 [:,:,:] split_validate,
        INT64 [:,:,:] split_train_and_validate,
        INT64 [:] i_start_by_x,  # (n_x,) kernel start index
        INT64 [:] i_stop_by_x, # (n_x,) kernel stop index
        double delta,
        double mindelta,
        int error,
        int selective_stopping,
        int num_threads,
):
    """Estimate multiple filters with boosting"""
    cdef:
        Py_ssize_t i_start = np.min(i_start_by_x)
        Py_ssize_t n_y = y.shape[0]
        Py_ssize_t n_x = x.shape[0]
        Py_ssize_t n_splits = split_train.shape[0]
        Py_ssize_t n_total = n_splits * n_y
        Py_ssize_t n_times = x.shape[1]
        Py_ssize_t n_times_h = np.max(i_stop_by_x) - i_start
        FLOAT64[:,:,:,:] hs = np.empty((n_splits, n_y, n_x, n_times_h))
        INT8[:,:] hs_failed = np.zeros((n_splits, n_y), 'int8')
        Py_ssize_t i, i_y, i_split
        BoostingRunResult *result
        INT8 * x_active
        FLOAT64 * y_error

    with nogil, parallel(num_threads=num_threads):
        y_error = <FLOAT64*> malloc(sizeof(FLOAT64) * n_times)
        x_active = <INT8*> malloc(sizeof(INT8) * n_x)

        for i in prange(n_total, schedule='guided'):
            i_y = i // n_splits
            i_split = i % n_splits
            result = boosting_run(y[i_y], x, x_pads, hs[i_split, i_y], split_train[i_split], split_validate[i_split], split_train_and_validate[i_split], i_start_by_x, i_stop_by_x, delta, mindelta, error, selective_stopping, i_start, n_times_h, x_active, y_error)
            hs_failed[i_split, i_y] = result.failed
            free_history(result)

        free(x_active)
        free(y_error)

    hs_failed_array = np.asarray(hs_failed)
    if np.any(hs_failed_array == 2):
        raise RuntimeError("Boosting: maximum number of iterations exceeded")
    return hs.base, hs_failed_array.astype(bool)


ctypedef struct BoostingStep:
    Py_ssize_t i_step
    Py_ssize_t i_stim
    Py_ssize_t i_time
    double delta
    double e_test
    double e_train
    BoostingStep *previous


ctypedef struct BoostingRunResult:
    int failed  # 0: success; 1: validation error never improved (h is not meaningful); 2: iteration limit exceeded
    BoostingStep *history


cdef BoostingRunResult * boosting_run_result(int failed, BoostingStep *history) noexcept nogil:
    result = <BoostingRunResult*> malloc(sizeof(BoostingRunResult))
    result.failed = failed
    result.history = history
    return result


cdef void free_history(
        BoostingRunResult *result,
) noexcept nogil:
    cdef:
        BoostingStep *step
        BoostingStep *step_i

    step = result.history
    while step != NULL:
        step_i = step.previous
        free(step)
        step = step_i
    free(result)


cdef BoostingRunResult * boosting_run(
        FLOAT64 [:] y,  # (n_times,)
        FLOAT64 [:,:] x,  # (n_x, n_times)
        FLOAT64 [:] x_pads,  # (n_x,)
        FLOAT64 [:,:] h,  # (n_x, n_times_h)
        INT64[:,:] split_train,  # Training data index
        INT64[:,:] split_validate,  # Validation data index
        INT64[:,:] split_train_and_validate,  # Training and validation data index
        INT64 [:] i_start_by_x,  # (n_x,) kernel start index
        INT64 [:] i_stop_by_x, # (n_x,) kernel stop index
        double delta,
        double mindelta,
        int error,
        int selective_stopping,
        Py_ssize_t i_start,
        Py_ssize_t n_times_h,
        # buffers
        INT8 * x_active,
        FLOAT64 * y_error,
) noexcept nogil:
    cdef:
        int out
        Py_ssize_t n_x = x.shape[0]
        Py_ssize_t n_x_active = n_x
        Py_ssize_t n_times = x.shape[1]
        BoostingStep *step
        BoostingStep *step_j
        BoostingStep *history
        # history
        double best_test_error = inf
        Py_ssize_t best_step_i = -1
        int n_bad, undo
        Py_ssize_t i_step, i

    # initialize buffers
    h[...] = 0
    for i in range(n_x):
        x_active[i] = 1
    for i in range(n_times):
        y_error[i] = y[i]

    # first step
    step = <BoostingStep*> malloc(sizeof(BoostingStep))
    step.previous = NULL
    step.i_step = -1
    step.i_stim = -1
    step.i_time = -1
    step.delta = 0
    step.e_train = error_for_indexes_c(y_error, split_train, error)
    history = step

    # pre-assign iterators
    for i_step in range(999999):
        # evaluate current h
        e_test = error_for_indexes_c(y_error, split_validate, error)
        step.e_test = e_test

        # evaluate stopping conditions
        if e_test < best_test_error:
            # print(' ', e_test, '<', best_test_error)
            best_test_error = e_test
            best_step_i = step.i_step
        elif e_test > step.previous[0].e_test and i_step >= 2:
            step_j = step.previous
            if selective_stopping:
                undo = 1
                if selective_stopping > 1:
                    # only stop if the predictor overfits n times without intermittent improvement
                    n_bad = 1
                    while True:
                        undo += 1
                        if step_j.i_stim == step.i_stim:
                            if step_j.e_test > step_j.previous[0].e_test:
                                # the same stimulus caused an error increase
                                n_bad += 1
                                if n_bad == selective_stopping:
                                    break
                            else:
                                undo = 0
                                break
                        step_j = step_j.previous
                        if step_j == NULL or step_j.e_test > e_test:
                            undo = 0
                            break  # the error improved

                if undo:
                    # disable predictor
                    x_active[step.i_stim] = 0
                    n_x_active -= 1
                    if n_x_active == 0:
                        break
                    # revert changes (skipping delta-halving steps, which did not change h)
                    for i in range(undo):
                        if step.delta:
                            h[step.i_stim, step.i_time] -= step.delta
                            update_error(y_error, x[step.i_stim], x_pads[step.i_stim], split_train_and_validate, -step.delta, step.i_time + i_start)
                        step_j = step.previous
                        free(step)
                        step = step_j
                    history = step
            # Basic
            # -----
            # stop the iteration if all the following requirements are met
            # 1. more than 10 iterations are done
            # 2. The testing error in the latest iteration is higher than that in
            #    the previous two iterations
            elif i_step > 10 and e_test > step_j.previous[0].e_test:
                # print("error(test) not improving in 2 steps")
                break

        # generate possible movements -> training error
        step = <BoostingStep *> malloc(sizeof(BoostingStep))
        step.previous = history
        step.i_step = i_step
        step.e_test = history.e_test  # valid until h is updated (i.e., for steps that are not applied)
        generate_options(step, y_error, x, x_pads, x_active, split_train, i_start, i_start_by_x, i_stop_by_x, error, delta, n_x)
        history = step

        # If no improvements can be found reduce delta
        if step.e_train > step.previous.e_train:
            # h is unchanged, so the step keeps the current training error (not the rejected candidate's)
            step.e_train = step.previous.e_train
            step.delta = 0
            step.i_stim = -1
            step.i_time = -1
            delta *= 0.5
            if delta >= mindelta:
                continue
            else:
                # print("No improvement possible for training data")
                break

        # abort if we're moving in circles
        if step.delta == -step.previous.delta and step.i_stim == step.previous.i_stim and step.i_time == step.previous.i_time:
            step.delta = 0
            # print("Moving in circles")
            break

        # update h with best movement
        h[step.i_stim, step.i_time] += step.delta
        update_error(y_error, x[step.i_stim], x_pads[step.i_stim], split_train_and_validate, step.delta, step.i_time + i_start)
    else:
        # exceptions can not propagate from a noexcept nogil function; the caller raises
        return boosting_run_result(2, history)

    # reverse changes after best iteration
    if best_step_i > -1:
        while step.i_step > best_step_i:
            if step.delta:
                h[step.i_stim, step.i_time] -= step.delta
            step = step.previous
        return boosting_run_result(0, history)
    else:
        # print(' failed')
        return boosting_run_result(1, history)


cdef void generate_options(
        BoostingStep * step,
        FLOAT64 * y_error,
        FLOAT64 [:,:] x,  # (n_x, n_times)
        FLOAT64 [:] x_pads,  # (n_x,)
        INT8 * x_active,  # for each predictor whether it is still used
        INT64 [:,:] indexes,  # training segment indexes
        int i_start,  # kernel start index (time axis offset)
        INT64 [:] i_start_by_x,  # (n_x,) kernel start index
        INT64 [:] i_stop_by_x, # (n_x,) kernel stop index
        int error,  # ID of the error function (l1/l2)
        double delta,
        Py_ssize_t n_x,
    ) noexcept nogil:
    cdef:
        double e_add, e_sub, e_new, x_pad
        Py_ssize_t i_stim, i_time, new_sign
        FLOAT64 * x_stim

    step.e_train = inf
    step.i_stim = -1
    step.i_time = -1
    step.delta = 0
    for i_stim in range(n_x):
        if x_active[i_stim] == 0:
            continue
        x_stim = &x[i_stim, 0]
        x_pad = x_pads[i_stim]
        for i_time in range(i_start_by_x[i_stim], i_stop_by_x[i_stim]):
            # +/- delta
            if error == 1:
                l1_for_delta(y_error, x_stim, x_pad, indexes, delta, i_time, &e_add, &e_sub)
            else:
                l2_for_delta(y_error, x_stim, x_pad, indexes, delta, i_time, &e_add, &e_sub)

            i_time -= i_start
            if e_add > e_sub:
                e_new = e_sub
                new_sign = -1
            else:
                e_new = e_add
                new_sign = 1

            # find smallest error
            if e_new < step.e_train:
                step.e_train = e_new
                step.i_stim = i_stim
                step.i_time = i_time
                step.delta = delta * new_sign


cdef void update_error(
        FLOAT64 * y_error,
        FLOAT64 [:] x,
        double x_pad,  # pad x outside valid convolution area
        INT64 [:,:] indexes,  # segment indexes
        double delta,
        Py_ssize_t shift,
    ) noexcept nogil:
    cdef:
        Py_ssize_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        if seg_start == -1:
            break
        seg_stop = indexes[seg_i, 1]
        conv_start = seg_start
        conv_stop = seg_stop
        if shift > 0:
            conv_start += shift
        elif shift < 0:
            conv_stop += shift
        # padding
        d = delta * x_pad
        # pre-
        for i in range(seg_start, conv_start):
            y_error[i] -= d
        # post-
        for i in range(conv_stop, seg_stop):
            y_error[i] -= d
        # part of the segment that is affected
        for i in range(conv_start, conv_stop):
            y_error[i] -= delta * x[i - shift]


cdef class Step:
    cdef readonly:
        long i_step, i_stim, i_time
        double delta, e_test, e_train

    def __init__(self, i_step, i_stim, i_time, delta, e_test, e_train):
        self.i_step = i_step
        self.i_stim = i_stim
        self.i_time = i_time
        self.delta = delta
        self.e_test = e_test
        self.e_train = e_train

    def __repr__(self):
        return f"{self.i_step:4}: {self.i_stim:4}, {self.i_time:4} {self.delta:+} --> {self.e_train:10f} / {self.e_test:10f}"


def boosting_fit(
        FLOAT64 [:] y,  # (n_times,)
        FLOAT64 [:,:] x,  # (n_x, n_times)
        FLOAT64 [:] x_pads,  # (n_x,)
        INT64 [:,:] split_train,
        INT64 [:,:] split_validate,
        INT64 [:,:] split_train_and_validate,
        INT64 [:] i_start_by_x,  # (n_x,) kernel start index
        INT64 [:] i_stop_by_x, # (n_x,) kernel stop index
        double delta,
        double mindelta,
        int error,
        int selective_stopping = 0,
):
    """Single model fit using boosting

    Parameters
    ----------
    y : array (n_times,)
        Dependent signal, time series to predict.
    x : array (n_x, n_times)
        Stimulus.
    x_pads : array (n_x,)
        Padding for x.
    split_train
        Training data index.
    split_validate
        Validation data index.
    split_train_and_validate
        Training and validation data index.
    i_start_by_x : ndarray
        Array of i_start for trfs.
    i_stop_by_x : ndarray
        Array of i_stop for TRF.
    delta : scalar
        Step of the adjustment.
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.
    error: int
        Error function to use (1 for l1, 2 for l2).
    selective_stopping : int
        Selective stopping.
    """
    cdef:
        BoostingRunResult *result
        BoostingStep *step
        Step step2
        Py_ssize_t i_start = np.min(i_start_by_x)
        Py_ssize_t n_x = x.shape[0]
        Py_ssize_t n_times = x.shape[1]
        Py_ssize_t n_times_h = np.max(i_stop_by_x) - i_start
        FLOAT64[:, :] h = np.empty((n_x, n_times_h))
        INT8 * x_active = <INT8 *> malloc(sizeof(INT8) * n_x)
        FLOAT64 * y_error = <FLOAT64 *> malloc(sizeof(FLOAT64) * n_times)

    x = np.asarray(x, order='C')
    result = boosting_run(y, x, x_pads, h, split_train, split_validate, split_train_and_validate, i_start_by_x, i_stop_by_x, delta, mindelta, error, selective_stopping, i_start, n_times_h, x_active, y_error)
    free(x_active)
    free(y_error)
    if result.failed == 2:
        free_history(result)
        raise RuntimeError("Boosting: maximum number of iterations exceeded")

    out = []
    step = result.history
    while step != NULL:
        step2 = Step(step.i_step, step.i_stim, step.i_time, step.delta, step.e_test, step.e_train)
        out.insert(0, step2)
        step = step.previous
    free_history(result)
    return np.asarray(h), out
