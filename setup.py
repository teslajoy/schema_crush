"""setup configuration for schema_crush package."""

from setuptools import setup, find_packages
from pathlib import Path

# read requirements (excluding git urls)
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file) as f:
    requirements = []
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("git+"):
            requirements.append(line)

# read readme
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="schema_crush",
    version="0.1.0",
    description="semantic schema matching using multi-agent embedders with human-in-the-loop validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nasim Sanati",
    author_email="nasim@plenary.org",
    url="https://source.ohsu.edu/Omicstra/schema_crush",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "schema_crush": [
            "data/**/*",
            "data/examples/*",
            "data/resources/**/*",
        ],
    },
    install_requires=requirements,
    python_requires=">=3.13",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.13"
    ],
)