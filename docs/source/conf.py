# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root to sys.path to enable importing protomotions modules
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ProtoMotions"
copyright = "2025, ProtoMotions Developers"
author = "ProtoMotions Developers"

version = ""  # No version number in docs
release = ""  # No version number in docs

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",  # Creates .nojekyll for GitHub Pages
    # 'sphinx_autodoc_typehints',  # Disabled - conflicts with mocked imports
    "sphinx_copybutton",
    "myst_parser",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Autosummary settings
autosummary_generate = True

# Mock imports for heavy dependencies that don't need to be installed for doc builds
# This allows Sphinx to parse docstrings without requiring GPU/simulation libraries

# Set environment variables before imports to avoid initialization issues
import sys  # noqa: E402
import os  # noqa: E402

# Prevent MuJoCo initialization issues
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


# Create a comprehensive mock for problematic imports
class Mock:
    """Mock class for imports that can't be satisfied."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    def __getattr__(self, name):
        if name in ("__file__", "__path__", "__version__"):
            return "/dev/null"
        if name == "__mro_entries__":
            # Return a function that returns a tuple for proper class inheritance
            return lambda bases: ()
        return Mock()

    def __getitem__(self, name):
        return Mock()

    def __iter__(self):
        return iter([])


# Install comprehensive mocks before any imports
# This must happen before autodoc tries to import any protomotions modules
mock_modules = [
    "mujoco",
    "mujoco.mjVERSION_HEADER",
    "tensordict",
    "pytorch_lightning",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.distributed",
    "torch.cuda",
    "torch.utils",
    "torch.utils.data",
    "lightning",
    "lightning.fabric",
    "lightning.fabric.fabric",
    "lightning_fabric",
    "isaacgym",
    "isaacgym.gymapi",
    "isaacgym.gymutil",
    "isaacgym.gymtorch",
    "isaacgymenvs",
    "genesis",
    "omni",
    "wandb",
    "hydra",
    "hydra.utils",
    "omegaconf",
    "tqdm",
    "trimesh",
    "pyvista",
    "smplx",
    "smpl",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.transform",
    "PIL",
    "cv2",
    "rich",
    "rich.progress",
    "skimage",
    "imageio",
    "openmesh",
    "gym",
    "easydict",
    "dm_control",
    "dm_control.mjcf",
    "dm_control.mujoco",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
]

for mod in mock_modules:
    sys.modules[mod] = Mock()


# Special handling for commonly imported items
class MockTensorDict(Mock):
    def __call__(self, *args, **kwargs):
        return Mock()


class MockTensor(Mock):
    def __call__(self, *args, **kwargs):
        return Mock()


class MockFabric(Mock):
    def __call__(self, *args, **kwargs):
        return Mock()


# Add specific class mocks
sys.modules["tensordict"].TensorDict = MockTensorDict
sys.modules["torch"].Tensor = MockTensor
sys.modules["torch"].nn = Mock()
sys.modules["lightning.fabric"].Fabric = MockFabric

autodoc_mock_imports = [
    "torch",
    "lightning",
    "lightning.fabric",
    "lightning_fabric",
    "pytorch_lightning",
    "tensordict",
    "isaacgym",
    "isaacgymenvs",
    "omni",
    "genesis",
    "wandb",
    "hydra",
    "omegaconf",
    "tqdm",
    "trimesh",
    "pyvista",
    "smplx",
    "smpl",
    "numpy",
    "scipy",
    "PIL",
    "cv2",
    "matplotlib",
    "rich",
    "skimage",
    "imageio",
    "openmesh",
    "gym",
    "easydict",
    "dm_control",
    "dm_control.mjcf",
    "dm_control.mujoco",
    "OpenGL",
    "OpenGL.GL",
    "pyopengl",
]

templates_path = ["_templates"]
exclude_patterns = []

# Exclude modules that execute code at module level
autodoc_exclude_modules = [
    "protomotions.train_agent",
    "protomotions.eval_agent",
    "protomotions.train_slurm",
    "protomotions.joint_monkey",
]

# Suppress specific warnings
suppress_warnings = [
    "autosummary",  # Suppress autosummary warnings
    "toc.not_included",  # Suppress warnings about documents not in toctree
    "toc.excluded",  # Suppress warnings about excluded documents
    "autodoc",  # Suppress autodoc warnings for mocked objects
    "autodoc.import_object",  # Suppress import warnings
]

language = "en"

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

# Global TOC depth - show all levels in navigation
html_show_sourcelink = False

# Ensure all toctrees contribute to global navigation
toc_object_entries_show_parents = "hide"

# Sidebar configuration - use global TOC everywhere
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ],
}

# Output file base name for HTML help builder.
htmlhelp_basename = "ProtoMotionsdoc"

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Custom event handlers ---------------------------------------------------


def setup(app):
    """Setup function called by Sphinx during initialization."""
    # Connect to autodoc events with error handling
    app.connect("autodoc-skip-member", skip_member_handler, priority=500)


def skip_member_handler(app, what, name, obj, skip, options):
    """
    Custom handler for autodoc-skip-member event.

    This handler fixes issues with mocked objects that don't have proper signatures.
    Returns True to skip the member, False to include it, None to use default behavior.
    """
    try:
        # Handle Mock objects gracefully
        if isinstance(obj, Mock):
            # Skip Mock objects unless they're explicitly documented
            return True

        # Don't skip members by default
        return None
    except Exception:
        # If there's any error in processing, don't skip
        return None
