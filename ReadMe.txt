This ReadMe covers the organization and use of the accompanying code.

The directory layout is as follows:

 - Relative Root
 	- data
 		- <Datasets>
 	- checkpoints
 		- <Saved model checkpoints from training>
 	- src
 		- logs
 			- <text outputs from experiment runtime>
 		- Scripts and .ipynb files for running experiments
 		- AuxiliaryScripts
 			- <Required Python scripts for experiments>


.ipynb Files:
	 The provided .ipynb file contains sample code for running the Two-Task and Six-Task experiments from the paper as well as setting up Tiny Imagenet.

Experiment Scripts:
	Main.py: Directs the sequential training of a network on a series of tasks. Used to train all experiments
	Eval Two Task Experiments:   Can collect a dictionary of all connectivitiesd or activations for a two-task experiment, and/or evaluate accuracy

Auxiliary Scripts:
	Manager.py: Manager is the primary class which orchestrates pruning, training, and weight sharing of the model
	Network.py: A class which holds the model and performs operations on the model such as switching out the classifier for each task or masking operations
	clmodels.py: Defines the VGG16 and Modified ResNet-18 networks used in this work
	cldatasets.py: Loads the necessary datasets. Includes the setup code for the MPC dataset (TIC setup requires an accompanying ipynb notebook).
	DataGenerator.py: A simple data generator for pytorch
	Utils.py: Performs miscellaneous functions including mask operations and activation collections.
	Hsic_estimator.py: The CKA estimator implemented for supplementary material experiments

