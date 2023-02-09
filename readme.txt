1. Test environment:
python: 3.8
tensorflow: 2.6.0
pandas: 1.3.3
numpy: 1.19.5


2. The description of each file:
dataset folder: where you put your data files inside. A sample dataset "/dataset/real/vic_2000_40_40.txt" is put for code running.
util.py: save the utility functions shared by the algorithms
global_var.py: save the global variables shared by the algorithms
run_a2c.py: save the functions of reinforcement learning algorithm.
a2c_env.py: save the environment for reinforcement learning.
a2c_brain.py: save the main functions of deep-q-network.


3. How to test
After installing the necessary software packages, run the file "run_evaluation.py". The parameter tuning can be done by modifying some variables inside the file.