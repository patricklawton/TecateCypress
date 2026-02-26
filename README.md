This repository contains code to reproduce the analysis and figures associated with the article "Robust allocation of fire management resources towards threatened species conservation under uncertainty" to be published in the journal Ecological Modelling.

Python package dependencies can be installed by creating a new conda environment using the file "environment.yml". 

To run the model fitting procedure using simulation-based inference, navigate to the "model_fitting" folder and run the following sequence of commands:
python sbi_main.py
python calibration.py
python lc2st.py

Then, to run the robustness analysis and generate figures, navigate to the "robustness" folder and run the following sequence of commands:
python init.py
python project.py run -o run_sims compute_lambda_s -p NUM_PROCS --progress
OPENBLAS_NUM_THREADS=1 mpiexec -np NUM_PROCS python phase.py
python make_figures.py
