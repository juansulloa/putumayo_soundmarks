# Putumayo soundmark detection

The objective of this repository is to provide a modular and flexible framework to fit predictive models able to detect automatically biotic and abiotic soundmarks in complex acoustic environments. The process follows four main steps: (1) annotate samples, (2) structure audio data, (3) tune model, (4) make inferences on new data.

## Dependencies

To run these experiments, it is highly recommended to work in a new virtual environment. The main dependencies are:

- Python 3.7
- Numpy
- Scipy
- Pandas
- scikit-image
- scikit-maad
- scikit-learn

All dependencies are listed in `./conda_environment/anfibia.yml`, a text file that can be used by Anaconda to create a new virtual environment with everything you need:

```
conda env create -f anfibia.yml
conda activate anfibia
```

## 1. Annotate samples

Select samples from the audio dataset (*corpus*) and annotate the presence of the soundmark of interest. The annotation can be done with Raven or Audacity (freeware).


## 2. Structure audio data

To feed the model with training data, it is necessary to structure the data correctly. 

- Discretize signal: We will first discretize the signal into short regions of interest (1-5 seconds) and assigned to a label
- Compute features: each region will be characterized using a bank of 2D wavelets.
- Data augmentation procedures are well suited to increase the training size and regularize the model.


## 3. Tune model

Use a systematic approach to improve the performance of the model.

Reduce avoidable bias
- Train bigger model
- Train longer or use better optimization algorithms
- Try other architectures, search for better hyperparameters

Reduce variance
- Get more data
- Regularization: L2, dropout, data augmentation
- Try other architectures, search for better hyperparameters

Carry out error analysis

References:
- Coursera: hyperparameter tuning
- Hands on machine learning (Aurélien Géron)
- Leslie 2018, A disciplined approach to neural network hyper-parameters: part 1 – learning rate, batch size, momentum, and weight decay.

## 4. Make inferences on new data

Inference on new data is performed with trained models found in the folder `./models`. You will also need to load the preprocessing options used to trained the model. The step by step procedure is found in the script `./scripts/model_predict.py`.