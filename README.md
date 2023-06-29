## Sentiment Analysis on Smile Twitter Dataset

This project focuses on utilizing a pretrained BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis using the Smile Twitter dataset. The BERT model is fine-tuned on the dataset to classify tweets into positive, negative, or neutral sentiment categories.
- Loading a pretrained BERT model is beneficial as it leverages transfer learning (providing a strong base of learned linguistic representations), especially when working with small datasets and under restricted computational power.
## Getting Started

To get started with this project, follow these steps:

1. **Set up a Python Environment**: Install the required dependencies mentioned in `requirements.txt`.
2. **Go through the train.py Script**: It provides a step-by-step recipe to execute the project.
    - **Summarized Steps**:
        1. **Data Preprocessing**: Load, clean and preprocess the data.
        2. **Train-Test Split**: Split the preprocessed data into training and validation sets.
        3. **Data Split Assignment**: Label the respective data splits in the DataFrame.
        4. **Data Encoding**: Tokenize the data for the BERT model.
        5. **TensorDatasets Creation**: Create PyTorch TensorDatasets for training and validation data.
        6. **Model Loading**: Load a pretrained BERT model.
        7. **DataLoaders Creation**: Create PyTorch DataLoaders for the training and validation datasets.
        8. **Optimizer and Scheduler Setup**: Set up optimizer and scheduler for model training.
        9. **Model Training**: Train the model using the training data.
        10. **Model Evaluation**: Evaluate the performance of the model on the validation dataset.

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









