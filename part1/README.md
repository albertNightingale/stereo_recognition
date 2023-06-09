# CS 6320 Project 5: [Depth Estimation using Stereo]

## Setup
1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyways.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj4_env_<OS>.yml`
3. This should create an environment named 'proj5'. Activate it using the Windows command, `activate proj5 or the MacOS / Linux command, `source activate proj5`
4. Install the project package, by running `pip install -e .` inside the repo folder.
5. Run the notebook using `jupyter notebook ./proj5_code/simple_stereo.ipynb`
6. Ensure that all sanity checks are passing by running `pytest` inside the "unit_tests/" folder.
7. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --username <your_uid>` and submit to Canvas (don't forget to submit your report to Gradescope!).
