# CoRe (/'koːre/) : Cosmological Regressor

A modular Python framework for performing non-parametric reconstructions of cosmological functions and propagating those results into derived cosmological quantities. This toolkit is designed for high-performance inference, with a **JAX** module for Gaussian Processes, a binning-focused class, and an interface to **CAMB** for theoretical cosmological quantities.

### Disclaimer

This code is a personal, work in progress pet-project. I cannot promise frequent updates or quick replies to issues and comments, but I plan to use this code for my work so it will not be abandoned, at least as long as it aligns with my research interests.

Feel free to use the code and report any bug or issues in this repository. If you use this code, please cite the first paper using it:
[Martinelli and Sapone (2026)](https://arxiv.org/abs/2606.16844)
```
@article{Martinelli:2026wjp,
    author = "Martinelli, Matteo and Sapone, Domenico",
    title = "{The cosmic tetrarchy: four estimators breaking the assumption degeneracy in cosmological distance tensions}",
    eprint = "2606.16844",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "6",
    year = "2026"
}
```

---

## Project Structure

### Folders

The source code is in the `CoRe` folder, whose internal structure is reported below

* **`Cosmology/`**: 
    * `cosmo.py`: Contains the `CalcCosmology` class. This handles the interface with **CAMB** to obtain theoretical cosmological quantities based on input parameters.
* **`derived_functions/`**: 
    * `reconstructor.py`: Defines the `DerivedFunction` class, used to transform reconstructed functions into physical cosmological observables.
* **`reconstruction_tools/`**: 
This folder is designed to be extensible for future methods (e.g. Splines, Neural Networks)
    * `GaussianProcess.py`: Core implementation of the `GPCalculator` class. This is the engine for GP reconstructions. 
    * `Binning.py`: CoRe implementation of the `BinnedCalculator` class. This class allows to bin observational data and obtain numerical derivatives and integrals.
* **`utils/`**: 
    * `diagnostics.py`: Contains the `Scorer` class for comparing reconstructions against data or theoretical models.
    * `sampler_interface.py`: A unified interface for samplers; currently supports the **Nautilus** nested sampler. More samplers will be added soon(ish)
* **`running_settings/`**: 
    * `Example.yaml`: A template configuration file for automated runs.

### Main Directory
* `runner.py`: The main entry point script. Execute reconstructions via:
    ```bash
    python runner.py running_settings/Example.yaml
    ```
* `DEMO-GaussianProcess.ipynb`: An example use of the `GPCalculator` class, including the mathematical foundations of Gaussian Processes.
* `DEMO-Binning.ipynb`: a guideline on the use of the `BinnedCalculator` class, containing also the mathematical description of the available approaches.
* `DEMO-derived_reconstruction.ipynb`: Walkthrough on obtaining derived functions from primary reconstructions.
* `DEMO-Scorer.ipynb`: Tutorial on using the `Scorer` class to evaluate reconstruction performance across different settings.

---

## Reconstruction Methodology

### Gaussian Process Reconstruction
The code contains a JAX based Gaussian Processes (GP) method to reconstruct functions without assuming a rigid parametric form. The `GPCalculator` provides:
* **MAP (Maximum A Posteriori)** optimization for hyperparameters.
* **Full Bayesian Inference** for hyperparameters via nested sampling.
* **Derivative/Integral Reconstruction**: Specifically designed to handle cosmological data where we often measure the derivative or integral of the underlying quantity.

We refere the reader to the notebook `DEMO-GaussianProcess.ipynb` for a more detailed discussion of the GP formalism.

### Binned reconstruction
The code allows to bin observational data and obtain numerical derivatives and integrals of the binned quantities. This is handled in the `BinnedCalculator` class, which provides two binning methods:
* **Flat binning** which divides the data by a user selcted fiducial and bins the residual, assuming fixed bin widths.
* **GLS binning** acting directly on the provided data and adjusting the bin width to optimize population

In addition, this class contains also a function obtaining numerical derivatives and integrals of the binned quantities.


We refer the reader to the notebook `DEMO-Binning.ipynb` for a more detailed discussion of this approach.

### Derived Functions
Once a function is reconstructed (e.g., $H(z)$), the `DerivedFunction` class allows you to propagate the uncertainties into secondary quantities (e.g., the luminosity distance $d_L(z)$) through either direct sampling or realization-based methods.

The code deals in a very automatized way with the reconstruction of derived functions. The notebook `DEMO-derived_reconstruction.ipynb` contains a walkthrough of the different methods that can be used.

---

## Getting Started

1.  **Installation**:
    Ensure you have the required dependencies:
    ```bash
    pip install jax jaxlib camb nautilus pandas numpy pyyaml getdist
    ```
    The code is pip installable and once downloaded it can be installed globally with
    ```bash
    pip install .
    ```

2.  **Running an Example**:
    Check the `running_settings/Example.yaml` to ensure the paths match your local setup, then run:
    ```bash
    python runner.py running_settings/Example.yaml
    ```

3.  **Exploration**:
    Open `DEMO-GaussianProcess.ipynb` to see how to manually interact with the classes and visualize the results.

---

## Future plans 
* Support for additional samplers (e.g., Cobaya, emcee).
* Integration of alternative reconstruction methods like Genetic Algorithms (GA).
* Expanded diagnostic plots in the `Scorer` class.

---

## 📝 License
[MIT]
