from setuptools import setup, find_packages

setup(
    name="core-regressor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "camb",
        "nautilus-sampler",
        "pandas",
        "numpy",
        "pyyaml",
        "getdist",
        #"bios"
    ],
    description="CoRe: Cosmological Regressor framework",
)
