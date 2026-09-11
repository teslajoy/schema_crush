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
    version="1.2.0",
    description="map biomedical schemas to FHIR using agentic AI with calibrated matchers as tools with MCP server support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nasim Sanati",
    author_email="nasim@plenary.org",
    url="https://github.com/teslajoy/schema_crush",
    license="MIT",
    project_urls={
        "Source": "https://github.com/teslajoy/schema_crush",
        "Documentation": "https://github.com/teslajoy/schema_crush/blob/main/docs/schema_crush_overview.md",
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "schema_crush": [
            "data/resources/*.yaml",
            "data/resources/*/*.json",
            "data/db/*.db",
            "data/calibrators/pkl/*.pkl",
        ],
    },
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "schema-crush-mcp=schema_crush.mcp.server:main",
        ],
    },
    python_requires=">=3.13",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.13"
    ],
)