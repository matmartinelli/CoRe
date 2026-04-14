# CoRe (/'koːre/) : Cosmological Regressor

A modular Python framework for performing non-parametric reconstructions of cosmological functions and propagating those results into derived cosmological quantities. This toolkit is designed for high-performance inference, leveraging **JAX** for Gaussian Processes and **CAMB** for theoretical consistency.

---

## Project Structure

### Folders
* **`Cosmology/`**: 
    * `cosmo.py`: Contains the `CalcCosmology` class. This handles the interface with **CAMB** to obtain theoretical cosmological quantities based on input parameters.
* **`derived_functions/`**: 
    * `reconstructor.py`: Defines the `DerivedFunction` class, used to transform reconstructed functions into physical cosmological observables.
* **`reconstruction_tools/`**: 
    * `GaussianProcess.py`: Core implementation of the `GPCalculator` class. This is the engine for GP reconstructions. This folder is designed to be extensible for future methods (e.g., Splines, Neural Networks).
* **`utils/`**: 
    * `diagnostics.py`: Contains the `Scorer` class for comparing reconstructions against data or theoretical models.
    * `sampler_interface.py`: A unified interface for samplers; currently supports the **Nautilus** nested sampler.
* **`running_settings/`**: 
    * `Example.yaml`: A template configuration file for automated runs.

### Main Directory
* `runner.py`: The main entry point script. Execute reconstructions via:
    ```bash
    python runner.py running_settings/Example.yaml
    ```
* `DEMO-GaussianProcess.ipynb`: A deep dive into the `GPCalculator` class, including the mathematical foundations of Gaussian Processes.
* `DEMO-derived_reconstruction.ipynb`: Walkthrough on obtaining derived functions from primary reconstructions.
* `GPscorer.ipynb`: Tutorial on using the `Scorer` class to evaluate reconstruction performance across different settings.

---

## Reconstruction Methodology

### Gaussian Process Reconstruction
The code contains a JAX based Gaussian Processes (GP) method to reconstruct functions without assuming a rigid parametric form. The `GPCalculator` provides:
* **MAP (Maximum A Posteriori)** optimization for hyperparameters.
* **Full Bayesian Inference** for hyperparameters via nested sampling.
* **Derivative/Integral Reconstruction**: Specifically designed to handle cosmological data where we often measure the derivative or integral of the underlying quantity.

We refere the reader to the notebood `DEMO-GaussianProcess.ipynb` for a more detailed discussion of the GP formalism.

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
