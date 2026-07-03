# Sphinx configuration of the tdls documentation site. The pages are
# written in Markdown (MyST); the Doxygen API reference is built
# separately (docs/Doxyfile) and attached under api/ at deployment.
import re
from pathlib import Path

project = "tdls"
author = "Tristan Chenaille"

# The version is read from the library header so it cannot drift.
_version_header = Path(__file__).resolve().parents[2] / "include/tdls/core/version.hpp"
release = re.search(r'TDLS_VERSION_STRING "([0-9.]+)"', _version_header.read_text()).group(1)
version = release

extensions = ["myst_parser"]
myst_enable_extensions = ["colon_fence"]

html_theme = "sphinx_rtd_theme"
html_title = "tdls"
html_show_copyright = False
html_show_sphinx = True
