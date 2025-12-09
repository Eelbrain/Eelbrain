.. _development:
.. _contributing:

************
Contributing
************

Eelbrain is hosted on `GitHub <https://github.com/Eelbrain/Eelbrain>`_.

Welcome to the Eelbrain Contributor's Guide! This page is a gentle introduction to contributing: 
it explains how to get involved, propose changes, and meet our expectations for testing, style, and documentation. 
This guide is meant to be useful for all contributors—if you spot gaps or have suggestions, please let us know through the issue tracker.

.. contents:: Table of Contents
   :depth: 2
   :local:


The Development Version
-----------------------

Development takes place on the ``main`` branch, while release versions are maintained on
branches called ``r/0.26`` etc. For further information on working with
GitHub see `GitHub's instructions <https://help.github.com/articles/fork-a-repo/>`_.

The repository contains a conda environment that contains everything needed to use Eelbrain except Eelbrain itself.
First, clone the repository (or your `fork <https://help.github.com/articles/fork-a-repo>`_), and change into the repository directory::

    $ git clone https://github.com/Eelbrain/Eelbrain.git
    $ cd Eelbrain

To generate the ``eeldev`` environment, use::

    $ mamba env create --file=env-dev.yml

The development version of Eelbrain can then be installed with ``pip``::

    $ mamba activate eeldev
    $ pip install -e .

On macOS, the ``$ eelbrain`` shell script to run ``iPython`` with the framework
build is not installed properly by ``pip``; in order to fix this, run::

    $ ./fix-bin

In Python, you can make sure that you are working with the development version::

    >>> import eelbrain
    >>> eelbrain.__version__
    'dev'


Opening Issues
--------------

Bug reports and feature requests are welcome on the `GitHub Issue Tracker <https://github.com/Eelbrain/Eelbrain/issues>`_. If you're unsure whether something is a bug or an enhancement, feel free to ask in an issue.

**Reporting Bugs**
    Effective bug reports help us fix issues faster. Please include:

    - **Minimal Reproducible Example**: A short, self-contained code snippet that demonstrates the error. This is crucial for verification.
    - **Version Information**: Run ``import eelbrain; print(eelbrain.__version__)`` and include the output.
    - **Traceback**: The complete error traceback text.
    - **Description**: A clear description of the expected behavior versus the actual behavior.

**Bug fixes**
    If you notice a bug and are able to fix it, we welcome a pull request.
    Ideally, include a test to preempt future regressions (see :ref:`dev-testing`).

**Feature Requests**
    If you are thinking about implementing a new feature, please **open an issue first** to discuss the design. This keeps efforts aligned with the roadmap and reduces the chance of duplicate work.

Pull Request Workflow
---------------------

We follow a standard Git workflow. For more details, see `GitHub's Pull Request documentation <https://docs.github.com/en/pull-requests>`_. Keep pull requests small and focused for faster reviews and easier merges; the steps below outline the recommended workflow.

0. **Create a Fork** of [Eelbrain](https://github.com/Eelbrain/Eelbrain).
1. **Create a Branch**: Create a new branch from ``main`` for each feature or fix.
2. **Commit Changes**: Make your changes and commit them.
3. **Test Locally**: Run the relevant tests (e.g., ``make test`` or targeted ``pytest`` invocations) before opening a pull request.
4. **Open a Pull Request (PR)**:
    - **Use Draft Mode**: Draft mode keeps work-in-progress safe from accidental merges and invites early feedback on your approach. See `GitHub's guide on draft pull requests <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`_.
    - **Ready for Review**: Switch the PR status to "Ready for Review" when the implementation is complete and local tests pass.
    - Link to any relevant issues or discussions in the PR description.


Git Best Practices
------------------

- **Thoughtful commits and PRs**: Group related changes together; avoid bundling unrelated work. Keep PRs small and coherent. Non-trivial PRs should receive at least two reviews, and even if permissions allow it, do not merge your own PRs.
- **Clear commit messages**: Write meaningful subjects that describe what changed and why (e.g., ``Website: Fix typos in homepage text``; ``README.md: Add link to Well-Understood paper``). Avoid vague messages like "Fix stuff" or "Update". Ensure the message matches the actual changes.
- **Credit co-authors**: When multiple people contribute to a commit, append ``Co-Authored-By: Name <email>`` for each contributor at the end of the commit message.
- **Syncing with main**: Rebase onto ``main`` regularly to keep a clean history (avoid merge commits where possible). Rebase when conflicts arise or when upstream changes affect stable artifacts (e.g., ``stable/``) to prevent CI failures.
- **Squashing**: For long or noisy histories that iterate on the same code, squash related commits into concise units before merging. Aim for commits that each represent a coherent change rather than a series of incremental fixes to the same feature.


Code Review Process
-------------------

The code review process is a collaborative effort to improve code quality.

**Expectations**
    A core developer will review your code for correctness, style, and API consistency. Apply feedback globally; if a reviewer notes an issue in one file, check if it exists elsewhere in your changes.

**Responding to Feedback**
    When you have addressed all reviewer comments:

    1. Push the new commits to your branch.
    2. **Leave a comment** on the PR (e.g., "Ready for re-review") or re-request a review via the GitHub UI. This explicitly notifies the maintainers that you are ready for the next round.


.. _dev-testing:

Testing and Validation
----------------------

Tests for individual modules are included in folders called ``tests``, usually
on the same level as the module.
To run all tests, run ``$ make test`` from the Eelbrain project directory.
On macOS, tests need to run with the framework build of Python;
if you get a corresponding error, run ``$ ./fix-bin pytest`` from the
``Eelbrain`` repository root.

**Validation Workflow**
    1. **Run tests locally first**: Start with ``make test`` and confirm it passes on your machine.
       See the [pytest docs](https://docs.pytest.org/en/stable/how-to/usage.html) on how to run individual tests when working on a specific fix.
    2. **CI Automation**: All pull requests trigger a Continuous Integration (CI) workflow that automatically runs the full test suite. CI blocks merges when tests fail.


Coding Style and Documentation
------------------------------

**PEP 8 Style and Formatting**
    To facilitate maintenance and consistent reviews, we follow the `PEP 8 style guide <https://peps.python.org/pep-0008/>`_.
    We recommend using tools to ensure compliance:

    - **flake8**: Run ``$ flake8 eelbrain`` from the project root to check for issues locally.
    - **autopep8**: Use this to automatically fix common style issues.
    - **Import Order**: Imports should be grouped in the following order: standard library, third-party libraries (e.g., numpy, matplotlib), and local Eelbrain imports.

**Consistent Naming and API Consistency**
    To make the library intuitive, we strive for consistency across the API:

    - **Naming**: Parameter names should be consistent with existing functions (e.g., use ``cmap`` for colormaps, not ``colorscale``).
    - **API Design**: Functions that handle data should accept data directly as a parameter (e.g., a ``y`` parameter for an ``NDVar``), analogous to existing plotting functions.

**Type Hinting**
    We use type hints in all function signatures (see :mod:`typing`; e.g., ``def my_function(y: NDVar) -> Figure:``).
    There's no need to duplicate type information in docstrings if it is already present in the signature; the signature is the source of truth.
    For more information see :mod:`typing`.

**Docstrings**
    - We follow the `numpydoc style <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ for all docstrings.
    - **Class Documentation**: Document a class's ``__init__`` parameters in the **class docstring**, not inside the ``__init__`` method itself.

**Documentation Format**
    The documentation is written in `ReStructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext>`_.


Scientific Software Requirements
--------------------------------

As a scientific computing library, Eelbrain aims to protect data integrity:

**Reproducibility**
    Visualizations and computational results should be reproducible across different platforms. Code should produce consistent results given the same input.

**Scientific Accuracy**
    Visualizations should accurately represent the underlying neuroscience data without artifacts. Coordinate systems and data transformations should be mathematically correct.

**Domain Compatibility**
    New features should integrate seamlessly with neuroscience workflows (e.g., support ``NDVar`` objects, work within Jupyter notebooks).

**Performance on Large Datasets**
    Neuroscience datasets are often large. Test your code with realistic data sizes to ensure performance remains acceptable.


Recommended Tools
-----------------

While not required, the following tools are used by the maintainers and can streamline development:

- **SourceTree**: A graphical frontend for git (`link <https://www.sourcetreeapp.com>`_).
- **PyCharm**: A powerful Python IDE that can handle formatting and testing (`link <https://www.jetbrains.com/pycharm>`_).
- **VS Code**: A lightweight, extensible editor with rich Python tooling support (`link <https://code.visualstudio.com>`_).
