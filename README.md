
CNN_final.ipynb: main notebook to run the model, including hyper parameter settings.

Data.ipynb: download data from kaggle to the remote server 

Dataset.py (preprocessing the data): ImageDataset Class; value function: calculate weights, mean and variance; dataset function: data splitting, transformation, normalisation.

Model.py: class CNN: cnn architecture; train_and_validation function; predict function: for the test results.

Plot.py: plot_metrics function for plotting loss, accuracy, and other metrics.

