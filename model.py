from config import Config
from transformers import BertForSequenceClassification

def load_model(label_dict):
    """
    Load a pretrained BERT model for sequence classification.

    Args:
        label_dict (dict): Dictionary mapping labels to indices.

    Returns:
        model: A BertForSequenceClassification model.
    """
    model = BertForSequenceClassification.from_pretrained(
        Config.pretrained_model_name,
        num_labels=len(label_dict),
        output_attentions=Config.output_attentions,
        output_hidden_states=Config.output_hidden_states,
    )
    return model
