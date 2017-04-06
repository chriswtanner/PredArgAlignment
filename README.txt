PURPOSE:
    performs cross-doc co-ref resolution for entities and events

HOW TO RUN:
    within src/ there are 2 bash scripts.  runLSTM_allDirs.sh is the one i invoke on a grid w/ GPUs, which submits several jobs (unique, complete runs of our program, just with different parameters per run).

    to invoke the program under any other environment, just:
        (1) pass all variables (including 'path') to runLSTM_1b.sh just like how we are doing within runLSTM_allDirs.sh

        NOTE: 'path' should point to the base directory (i.e., wherever PredArgAlignment/ resides)
	
OUTPUT:
    the program will output the test set's F1 performance of EVENT coref after every 5 iterations of training.

    NOTE: for way more detailed information of what is going on, simply change the 'isVerbose' flag manually which resides at the top in:
        - Test.py (the main entry point of the program)
        - multilayer_perceptron.py (the FFNN classifier)
      
