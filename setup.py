import setuptools
import pathlib # Use pathlib for robust path handling

# Get the path to the directory containing setup.py
# This is typically the root of your project
here = pathlib.Path(__file__).parent.resolve()

# Construct the full path to README.MD relative to 'here' with correct casing
# Your directory structure shows README.MD (uppercase D), so we use that.
readme_path = here / "README.MD" 

# Read the long description from README.MD
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()

__version__="0.0.0"

REPO_NAME = "End-to-End-Machine-Learning-Project-with-MLFlow"
AUTHOR_USER_NAME = "tanayatipre8"
SRC_REPO = "MLProject"
AUTHOR_EMAIL = "tanayatipre8@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
