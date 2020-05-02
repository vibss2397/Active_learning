# Active_learn

The 64x64 greyscale images used for testing and training are in the "dataset" folder.
The "flatten_images.m" file takes these images and creates a "dataset.mat" file with training/test data/labels.
-The "active_learn.m" file executes active learning with AdaBoost, which can use decision trees, neural networks, or convolutional neural networks as weak learners.
 * The "train_boosted_dt.m" file trains AdaBoost, and "test_boosted_dt.m" tests it.
  * The "neural_train.m" file trains a neural network, and "predict_net.m" gives the network's prediction.
    * The "cnn.m" file trains a convolutional neural network.

-The "a_l.m" file executes active learning with the "cnn.m" learner.
