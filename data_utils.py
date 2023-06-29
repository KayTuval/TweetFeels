import pandas as pd
import torch

from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from config import Config


def preprocess_data():
    # Load the data from the specified data path
    df = pd.read_csv(Config.data_path,
                     names=['id', 'text', 'category'])
    df.set_index('id', inplace=True)

    # Replace 'not-relevant' with 'neutral' in the 'category' column
    df['category'] = df['category'].replace('not-relevant', 'neutral')

    # Filter the dataset to include only the desired emotion labels
    desired_labels = ["happy", "angry", "neutral", "surprise", "sad", "disgust"]
    df = df[df['category'].isin(desired_labels)]

    # Map the emotion labels to numeric indices
    possible_labels = df.category.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    df['label'] = df.category.replace(label_dict)

    return df, label_dict


def assign_data_split(df, train_indices, val_indices):
    """
    Assigns 'train' and 'val' values to the 'data_split' column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        train_indices (list): List of indices representing the training samples.
        val_indices (list): List of indices representing the validation samples.

    Returns:
        pd.DataFrame: The updated DataFrame with 'data_split' values assigned.
    """
    df['data_split'] = ['not_set'] * df.shape[0]  # Initialize the 'data_split' column

    # Assign 'train' to rows specified by train_indices
    df.loc[train_indices, 'data_split'] = 'train'

    # Assign 'val' to rows specified by val_indices
    df.loc[val_indices, 'data_split'] = 'val'

    return df

def encode_data(df):
    """
    This function expects a pandas DataFrame that includes the following columns:
    'id' (set as the index), 'text', 'category', 'label', and 'data_split'.
    - 'id': Unique identifier for each sample
    - 'text': The text data to encode
    - 'category': The category or class each sample belongs to
    - 'label': The label corresponding to the category (usually numerical)
    - 'data_split': Specifies which data split ('train' or 'val') each sample belongs to
    """

    # Ensure the DataFrame has the necessary columns
    required_columns = ['text','category', 'label', 'data_split']
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"DataFrame is missing the following required columns: {missing_columns}")

    # Initialize a BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained(
        Config.pretrained_model_name,
        do_lower_case=Config.do_lower_case
    )

    # Tokenize and encode the training data
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_split=='train'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=Config.pad_to_max_length,
        max_length=Config.max_length,
        return_tensors='pt'
    )

    # Tokenize and encode the validation data
    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_split=='val'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=Config.pad_to_max_length,
        max_length=Config.max_length,
        return_tensors='pt'
    )

    # Extract the encoded input IDs, attention masks, and labels from the encoded data
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_split=='train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_split=='val'].label.values)

    # Return a dictionary containing the input IDs, attention masks, and labels for each split.
    return {
        "train": {
            "input_ids": input_ids_train,
            "attention_masks": attention_masks_train,
            "labels": labels_train
        },
        "val": {
            "input_ids": input_ids_val,
            "attention_masks": attention_masks_val,
            "labels": labels_val
        }
    }

def create_dataloaders(dataset_train, dataset_val):
    """
    Create DataLoaders for the training and validation datasets.

    Args:
        dataset_train (TensorDataset): The training dataset.
        dataset_val (TensorDataset): The validation dataset.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.

    Raises:
        TypeError: If the input datasets are not instances of TensorDataset.
        ValueError: If the input datasets are empty.
    """
    # Check if the input datasets are instances of TensorDataset
    if not isinstance(dataset_train, TensorDataset) or not isinstance(dataset_val, TensorDataset):
        raise TypeError("Input datasets must be instances of TensorDataset.")

    # Check if the input datasets are empty
    if len(dataset_train) == 0 or len(dataset_val) == 0:
        raise ValueError("Input datasets must not be empty.")

    try:
        dataloader_train = DataLoader(
            dataset_train,
            sampler=RandomSampler(dataset_train),
            batch_size=Config.batch_size
        )

        dataloader_val = DataLoader(
            dataset_val,
            sampler=RandomSampler(dataset_val),
            batch_size=Config.val_batch_size
        )

        return dataloader_train, dataloader_val

    except Exception as e:
        print(f"An error occurred while creating the DataLoaders: {e}")
        raise

