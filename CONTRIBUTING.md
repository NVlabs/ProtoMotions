# How to Contribute

We started ProtoMotions3 to promote open research and help all members of the community. All participation in this project is open and not bound to corporate affiliation. 

## Automated Checks

Linting, formatting, license headers and spell checking all run through
[pre-commit](https://pre-commit.com/). Install the hook once and every commit is checked
before it is created:

```bash
pip install pre-commit
pre-commit install
```

**Run these tools through pre-commit rather than invoking them yourself.**
`.pre-commit-config.yaml` pins an exact version of each tool, and pre-commit installs
those versions into an isolated environment of its own. A `ruff` that happens to be on
your `PATH` is almost certainly a different version, and the formatter's output changes
between releases — so a file your local `ruff` wants to reformat may be one the pinned
version already considers correct. Going through pre-commit is what makes everyone's
result the same, and the same as CI's.

To check specific files without committing:

```bash
pre-commit run --files <file1> <file2> ...
```

Prefer that over `pre-commit run --all-files`: parts of the tree predate these checks, so
running over everything produces a large diff unrelated to your change.

## Releasing models

Code remains Apache-2.0. NVIDIA-provided model releases default to the NVIDIA
Open Model License unless explicitly approved otherwise. Follow the
[model-release checklist](legal/MODEL-LICENSING.md): include a License section
in the model card, the complete license and an attribution notice with every
model package, including exported models and adapters. Document any exception
in the card and the reviewed exception manifest. Run
`python scripts/check_model_licenses.py` before submitting a model release.

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

#### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
