import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("neural_lam/__init__.py", "r") as fv:
    for l in fv.readlines():
        if "__version__" in l:
            current_version = l.split('"')[1]
            break

setuptools.setup(
    name="neural_lam",
    version=current_version,
    author="Joel Oskarsson (original author), Thomas Rieutord (contributor to this fork), and many others",
    author_email="thomas.rieutord@met.ie",
    description="""Graph-based neural weather prediction models for Limited Area Modeling""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasRieutord/neural-lam",
    packages=setuptools.find_packages(),
    classifiers=(
        "Environment :: Console" "Programming Language :: Python :: 3",
        "Operating System :: Linux",
        "Development Status :: 2 - Pre-Alpha",
    ),
)
