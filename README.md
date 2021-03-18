# Quantum SLIM - Thesis
Repository of the thesis: **a Quantum approach of learning-based collaborative filtering 
model in Recommender System**.

The repository folders are structured as such:
 - **course_lib**: it contains the python library developed by Maurizio Ferrari Dacrema for the Polimi 
   RecSys of 2019 with few tweaks on "DataSplitter_Warm_k_fold.py" (for a bug fix) 
   and <"metrics.py", "Evaluator.py"> (for the correct version of item coverage).
 - **data**: it is the folder on which the dataset should be. Only csv dataset with a specific format
   are compatible with the experiments (i.e. csv file with 3 columns: first is the user id, second is
   the item id and third is the rating value)
 - **test**: it contains few tests done on the quantum model
 - **src**: it contains the source code of the quantum model with other utilities classes and functions
 - **experiments**: it contains the script code of the experiments. The important one is the folder of 
   experiments that contains all the "run" experiments used.

## Instructions
First of all, the necessary libraries for the environments are stored in "requirements.txt":

`pip install -r requirements.txt`

Then, remember to add this project in the PYTHONPATH environment if you run the experiments 
on terminal.

`export PYTHONPATH=$PYTHONPATH:/path/to/project/folder`

Finally, it is possible to run the experiments.
 1) For running a single experiment of quantum SLIM model, use "run_quantum_slim.py". For the parameters 
    pass, run the helper message:
    
    `python run_quantum_slim.py -h`
    
 2) For running a parameter tuning procedure, use "run_parameter_search_qslim.py" to run it for quantum SLIM model.
    Meanwhile, for the other CF models, use "run_parameter_search_cf.py". But first change the variable "MODEL" in the code"
 3) For running multiple experiments of quantum SLIM model, use "run_multiple_quantum_slim.py". For the set of 
    hyper-paremeters to test, you need to modify the dictionary in the script itself. This will run all the 
    combinations defined in the dictionary.
 4) In case, a crash has occurred in an experiment, you can resume it by running "resume_quantum_slim.py". Just
    pass the name of the folder in "report/quantum_slim/" folder.