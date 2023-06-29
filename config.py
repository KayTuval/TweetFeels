import torch

class Config:
## Training parameters
    num_workers = 0
    test_size= 0.15

    # Data
    data_path = "data/smile-annotations-final.csv"

    # Tokenizer and encoding parameters
    pretrained_model_name = 'bert-base-uncased'
    do_lower_case = True
    max_length = 256
    pad_to_max_length = True


    # Model configuration
    output_attentions = False
    output_hidden_states = False

    # DataLoader configuration
    batch_size = 4
    val_batch_size = 32

    # Optimizer and scheduler configuration
    lr = 1e-5
    eps = 1e-8
    epochs = 10

    # Metric configuration
    f1_score_average = 'weighted'


    # Random seed configuration
    random_state = 17

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def __repr__(self):
        msg = ''
        for key in dir(self):
            if not str(key).startswith('__'):
                msg += str(key) + '=' + str(getattr(self, key)) + '\n'
        return msg