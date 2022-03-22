Using Orientation to Distinguish Overlapping Chromosomes

Entry points:
semantic_segmentation.py to train the semantic segmentation models
semantic_segmentaiton_evaluation.py to evaluate pre-trained models
instance_segmentation.py to train the neural network that is used in the orientation-based segmentation
instance_segmentation_evaluation.py to evaluate the neural network that is used in the orientation-based segmentation
optimise_clustering.py was used to perform hyperparameter optimisation

networks.py implements the deep learning models
clusteirng.py contains the code for performing the post-processing of the orientation-based segmentation
datasets.py contains the code for using the datasets. Its main function is used for visualising the datset.
