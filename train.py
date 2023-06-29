import os
import torch
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from config import Config
from data_utils import *
from model import *
from train_utils import *
from metrics import *
import random



# Preprocess the data and get the preprocessed DataFrame
# The label_dict will be used for the model: num_labels.
df, label_dict = preprocess_data()

# Training/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size= Config.test_size,
    random_state=Config.random_state,
    stratify=df.label.values,  #The stratify strategy ensures that each split contains a proportional representation of the different labels present in the original dataset.
)

# Assign 'train' and 'val' values to the 'data_split' column
df = assign_data_split(df, X_train, X_val)

# Encode the data
encoded_data = encode_data(df)

# Create TensorDatasets for the training and validation datasets
dataset_train = TensorDataset(encoded_data['train']['input_ids'],
                              encoded_data['train']['attention_masks'],
                              encoded_data['train']['labels'])

dataset_val = TensorDataset(encoded_data['val']['input_ids'],
                            encoded_data['val']['attention_masks'],
                            encoded_data['val']['labels'])

print(f"len dataset_train: {len(dataset_train)}")
print(f"len dataset_val: {len(dataset_val)}")

# Load the model
model = load_model(label_dict)

# Create the DataLoaders
dataloader_train, dataloader_val = create_dataloaders(dataset_train, dataset_val)

# Set up the optimizer and scheduler
optimizer, scheduler = setup_optimizer_scheduler(model, dataloader_train)

# Create the "Models" directory if it doesn't exist (for saving model weights)
os.makedirs("Models", exist_ok=True)

# Setting random seeds allows reproducibility
random.seed(Config.random_state)
np.random.seed(Config.random_state)
torch.manual_seed(Config.random_state)
torch.cuda.manual_seed_all(Config.random_state)

# Train the model
# It is recommended to use Google colab GPU in case you don't have GPU available.
average_train_loss = train(model, dataloader_train, optimizer, scheduler)

# Load the "last" model weights
# (You may change the "last" to be any epoch as long that it is inside the Models directory).
model.load_state_dict(
            torch.load('Models/BERT_ft_last.model',
            map_location=torch.device('cpu')))

# Evaluate the model
_, predictions, true_vals = evaluate(model, dataloader_val)
accuracy_per_class(predictions, true_vals, label_dict)

