# Overview
This is the code to perform Coble creep deformation and void nucleation/growth simulation in a 3D polycrystalline solid.

# Requirements
* [Neper](https://neper.info/index.html) to generate initial grain distribution
  * Runs on Unix-like system (including Windows Subsystem for Linux)
  * [Installation guide](https://neper.info/doc/introduction.html#installing-neper) is available
 
* [Python 3.10](https://www.python.org/downloads/) to run the simulation
  
* Pandas, NumPy, Math, JSON, Random, SymPy, Scipy, and Workbook
  * To install these libraries, use the following commands:  
    `pip install pandas numpy math jsonlib random sympy scipy Workbook`

# How to run
1. Generate grain distribution using Neper Tessellation Module
   * We provided sample Neper tess file (`sample.tess`).
2. Set the parameters
   * Edit the values in the `parameter setting.txt` file to set the parameter values.
3. Save `sample.tess`, `parameter setting.txt` and `creep.py` in the same directory and execute `creep.py`. Strain and void area fraction data are exported in excel file.
