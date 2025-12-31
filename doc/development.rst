.. _development:
.. _contributing:

***********
Development
***********

Welcome to the Eelbrain Contributor's Guide! This page is a gentle introduction to contributing:
it explains how to get involved, report issues, and propose changes,
and how we handle testing, style, and documentation.
This guide is meant to be useful for all contributors—if you spot gaps or have suggestions, please let us know through the issue tracker.

.. contents:: Table of Contents
   :depth: 2
   :local:


The Development Version
-----------------------

Eelbrain is hosted on `GitHub <https://github.com/Eelbrain/Eelbrain>`_.
Development takes place on the ``main`` branch, while release versions are maintained on
branches called ``r/0.40`` etc. For further information on working with
GitHub see `GitHub's instructions <https://help.github.com/articles/fork-a-repo/>`_.

The repository contains a mamba environment that contains everything needed to use Eelbrain except Eelbrain itself.
To install the development version:
First, clone the repository (or your `fork <https://help.github.com/articles/fork-a-repo>`_), and change into the repository directory::

    $ git clone https://github.com/Eelbrain/Eelbrain.git
    $ cd Eelbrain

Generate the ``eeldev`` environment::

    $ mamba env create --file=env-dev.yml

The development version of Eelbrain can then be installed with ``pip``::

    $ mamba activate eeldev
    $ pip install -e .

On macOS, the ``$ eelbrain`` shell script to run ``iPython`` with the framework
build is not installed properly by ``pip``; in order to fix this, run::

    $ ./fix-bin

In Python, you can make sure that you are working with the development version::

    $ python
    >>> import eelbrain
    >>> eelbrain
    <module 'eelbrain' from '/Users/me/Code/Eelbrain/eelbrain/__init__.py'>


Opening Issues
--------------

Bug reports and feature requests are welcome on the GitHub `Issue Tracker <https://github.com/Eelbrain/Eelbrain/issues>`_.
If you're unsure whether something is a bug or an enhancement, feel free to ask in an issue.
For other questions, consider using GitHub `Discussions <https://github.com/Eelbrain/Eelbrain/discussions>`_.

**Reporting Bugs**
    Please open a bug-report `Issue <https://github.com/Eelbrain/Eelbrain/issues>`_ and include as much information as possible.
    Effective bug reports help us fix issues faster.

**Bug fixes**
    If you notice a bug and are able to fix it, we welcome a pull request.
    Ideally, include a test to preempt future regressions (see :ref:`dev-testing`).

**Feature Requests**
    If you are thinking about implementing a new feature, please open an issue first to discuss the design.
    This keeps efforts aligned with the roadmap and reduces the chance of duplicate work.


Pull Request Workflow
---------------------

We follow a standard GitHub `workflow for Pull Requests <https://docs.github.com/en/pull-requests>`_ (PRs).

.. IMPORTANT::
   **Keep PRs small and focused**: Each PR should address a single issue or feature.
   This allows for faster turnaround, which makes reviewing easier and minimizes the chance for code drift between your PR and the ``main`` branch.

The steps below outline the recommended workflow.

1. **Create a Fork** of `Eelbrain <https://github.com/Eelbrain/Eelbrain>`_.
2. **Create a Branch**: Create a new branch from ``main`` for each feature or fix.
3. **Commit Changes**: Make your changes and commit them.
4. **Test Locally**:
   Add tests for new features and bug fixes to ensure code quality and prevent regressions.
   Run existing tests to make sure nothing breaks (see :ref:`dev-testing`).
5. **Push to Your Fork**: Push your branch to your fork on GitHub.
6. **Open a Pull Request (PR)**:

   - **Use Draft Mode**: `Draft mode <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`_
     allows you to make sure all checks pass before inviting feedback.
   - Give the PR a descriptive title and describe the intended change.
     `Link to relevant issues <https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue>`_ and discussions in the PR description.
   - Monitor the checks, including:

       - Tests: in case of failures, you can inspect the errors that occurred.
       - Codecov: tells you whether tests cover your changes sufficiently.
       - Docs: if you modified the documentation, ensure it builds as intended.

   - **Ready for Review**: Switch the PR status to "Ready for Review" when the implementation is complete and all checks pass.
     Invite reviews from specific contributors (only if they are specifically relevant or have previously agreed to review this PR).

7. **Code Review**: The code review process is a collaborative effort to improve code quality.

   - Apply feedback globally; if a reviewer notes an issue in one file, check if it exists elsewhere in your changes.
   - When you have addressed all reviewer comments and checks are passing, signal that the PR is ready for another review by
     leaving a comment (e.g., "All comments addressed, ready for re-review") or
     re-requesting a review from the contributors whose comments you addressed, via the GitHub UI.
     This explicitly notifies the maintainers that the code is ready for the next round.
   - Try to address reviews promptly.
     Re-review is easier when changes are fresh in mind.
     in addition, the longer you wait, the more likely your changes will drift from the current codebase.


.. _dev-testing:

Testing and Validation
----------------------

Tests are embedded throughout the codebase in ``test`` folders.
Tests for individual modules are included in folders called ``tests``, usually
on the same level as the module.

Running tests locally (from the project root)::

    $ make test                               # runs all tests
    $ make test-no-gui                        # runs tests that do not invoke GUIs
    $ make test-only-gui                      # runs only the tests that invoke GUIs
    $ pytest path/to/test_file.py             # runs all tests in a specific file
    $ pytest path/to/test_file.py::test_func  # runs test_func() in path/to/test_file.py

For more options, see the `pytest docs <https://docs.pytest.org/en/stable/how-to/usage.html>`_.

An additional test for the :class:`MneExperiment` pipeline takes longer to run and can be run separately as needed::

    $ pytest --runslow eelbrain/_experiment/tests/test_sample_experiment.py::test_sample_source

All pull requests trigger a Continuous Integration (CI) workflow that automatically runs the full test suite.

On macOS, tests involving GUIs need to run with the framework build of Python;
if you get a corresponding error, run ``$ ./fix-bin pytest`` from the
``Eelbrain`` repository root.


Coding Style and Documentation
------------------------------

**PEP 8 Style and Formatting**
    To facilitate maintenance and consistent reviews, we follow the `PEP 8 style guide <https://peps.python.org/pep-0008/>`_.
    We recommend using tools to ensure compliance:

    - IDEs like PyCharm and VSCode can fix and alert you to style issues as you code.
    - **flake8**: Run ``$ flake8 eelbrain`` from the project root to check for issues locally.
    - **autopep8**: Use `this <https://pypi.org/project/autopep8/>`_ to automatically fix common style issues.

**Consistent Naming and API Consistency**
    To make the library intuitive, we strive for consistency across the API:

    - **Naming**: Parameter names should be consistent with existing functions (e.g., use ``cmap`` for colormaps, not ``colorscale``).
    - **Avoid abbreviations**: For new parameters, we generally prefer avoiding abbreviations, as they can be harder to remember.

**Type Hints**
    We use type hints in all function signatures (see :mod:`typing`; e.g., ``def my_function(y: NDVar) -> Figure:``).

**Docstrings**
    - We follow the `numpydoc style <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ for docstrings.
    - There's no need to duplicate type information in docstrings if it is already present in the signature; the signature is the source of truth.

**Documentation Format**
    - The documentation is written with :mod:`sphinx` in `ReStructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext>`_.


Recommended Tools
-----------------

For more on Git we recommend the free `Pro Git book <https://git-scm.com/book>`_.

The following tools are used by the maintainers and can streamline development:

- **SourceTree**: A graphical frontend for git (`link <https://www.sourcetreeapp.com>`_).
- **PyCharm**: A powerful Python IDE that can handle formatting and testing (`link <https://www.jetbrains.com/pycharm>`_).
- **VS Code**: A lightweight, extensible editor with rich Python tooling support (`link <https://code.visualstudio.com>`_).