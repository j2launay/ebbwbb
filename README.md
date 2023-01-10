# Explaining a Black Box without another Black Box
This repository contains the code of the paper Explaining a Black Box without another Black Box, and the counterfactuals methods: Growing Net and Growing Language.

The code to generate a counterfactual is available in counterfactuals_methods.py while function used for experiments are in text_function_experiments.py

k_closest_experiments.py, computer_outlierness_recall, and robustness_evaluation.py contain the code to launch the results obtained for sedc, Growing Net, and Growing Language respectively in Figure 2, 3, and 4.

The average recall from the Table 1 can be obtain from the code in compute_outlierness_recall.py for sedc, Growing Net, and Growing Language.

Results from Figure 5 to 9 are obtained from compute_minimality.py

Results from Figure 10 to 12 can be obtain in compute_outlierness_recall.py

run: pip install -r requirements.txt in your shell in order to install all the packages that are necessary to launch the experiments.
