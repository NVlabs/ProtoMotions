Contributing
============

Welcome! We're excited you're interested in contributing to ProtoMotions.

This project thrives on community involvement. Whether you're fixing a typo, adding a feature, or sharing your research—every contribution matters.

Ways to Contribute
------------------

There are many ways to help, and you don't need to be an expert:

**Share Your Work**
   Used ProtoMotions in a project or paper? Let us know! We'd love to feature it.

**Report Issues**
   Found a bug? Something confusing in the docs? `Open an issue <https://github.com/NVLabs/ProtoMotions/issues>`_ and tell us about it.

**Improve Documentation**
   Spotted a typo? Have a better explanation for something? Documentation improvements are always welcome.

**Add Examples**
   Created a cool experiment configuration or training setup? Share it with others.

**Fix Bugs**
   Browse `open issues <https://github.com/NVLabs/ProtoMotions/issues>`_ and pick one that interests you.

**Add Features**
   Have an idea for a new feature? Open an issue to discuss it first, then submit a PR.

Getting Started
---------------

1. **Fork the repository** on GitHub

2. **Clone your fork**:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/ProtoMotions.git
      cd ProtoMotions

3. **Create a branch** for your changes:

   .. code-block:: bash

      git checkout -b my-contribution

4. **Make your changes** and test them

5. **Commit** with a clear message (and sign-off):

   .. code-block:: bash

      git commit -s -m "Add feature X"

6. **Push** to your fork:

   .. code-block:: bash

      git push origin my-contribution

7. **Open a Pull Request** on GitHub

That's it! We'll review your PR and work with you to get it merged.

Signing Your Commits
--------------------

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license.

To sign off on a commit, use the ``--signoff`` (or ``-s``) option:

.. code-block:: bash

   git commit -s -m "Add cool feature"

This appends the following to your commit message:

.. code-block:: text

   Signed-off-by: Your Name <your@email.com>

**Note:** Any contribution which contains commits that are not signed-off will not be accepted.

Code Style
----------

We use `pre-commit <https://pre-commit.com/>`_ to automatically format and lint code. To set it up:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

This will run the following checks on every commit:

* **Ruff** for linting and formatting (replaces black, isort, flake8)
* **Typos** for spell checking
* License header insertion
* Trailing whitespace and end-of-file fixes

You can also run the checks manually:

.. code-block:: bash

   pre-commit run --all-files

Beyond automated formatting:

* Use meaningful variable and function names
* Add docstrings to new functions and classes
* Keep commits focused—one logical change per commit

Don't worry about being perfect. We're happy to help you polish your contribution during review.

Pull Request Tips
-----------------

* **Keep PRs focused**: One feature or fix per PR is easier to review
* **Describe your changes**: Help reviewers understand what you did and why
* **Test your changes**: Make sure things work before submitting
* **Be patient**: We'll get to your PR as soon as we can

Questions?
----------

Not sure where to start? Have questions about the codebase?

* Open a `GitHub Discussion <https://github.com/NVLabs/ProtoMotions/discussions>`_
* Check existing issues for similar questions
* Reach out in your PR if you get stuck

We're here to help. Don't hesitate to ask!

Thank You
---------

Every contribution—big or small—helps make ProtoMotions better for everyone. Thank you for being part of this project!


