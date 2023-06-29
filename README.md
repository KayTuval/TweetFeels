# BERT Sentiment Analysis on Smile Twitter Dataset

This project focuses on training a BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis using the Smile Twitter dataset. The BERT model is fine-tuned on the dataset to classify tweets into positive, negative, or neutral sentiment categories.

## Project Structure

The project is organized as follows:

- **data**: 
The Twitter SMILE dataset was downloaded from
"https://www.kaggle.com/datasets/ashkhagan/smile-twitter-emotion-dataset?resource=download.
- **data_utils.py**: Includes utility scripts that provide functions to preprocess the data, encode it create the data loaders, etc.
- **model.py**: Includes the pretrained BERT model for sequence classification.
- **train_utils.py**: Includes utility scripts that provide functions to set up the optimizer and scheduler, training loop and evaluate the modle.
- **metrices.py**: Includes functions to calculate f1 score, accuracy per class.
- **train.py**: The main script to run the project. 
- **Config.py**: contains settings and parameters for the project.
- **rqruirements.txt**

- **README.md**: Provides an overview of the project, dataset details, potential use cases, and information about citing and licensing the dataset.

## Getting Started

To get started with this project, follow these steps:

1. Set up a Python environment with the required dependencies mentioned in `requirements.txt`.

2. Download Smile Twitter dataset (`smile_tweets.csv`) and place it in the `data` directory.

3. Go through the training.py script. It is a step-by-step recipe to execute the project.




