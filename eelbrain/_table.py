    xs = [np.empty((n_cases, n_samples)) for _ in data_vars]
    for i, case_index in enumerate(case_indexes):
        try:
            for j, sample_index in enumerate(samples_indexes):
                index = np.logical_and(case_index, sample_index)