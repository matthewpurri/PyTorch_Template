import argparse

def check_arguments(args):
    pass

def save_config_file(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    # Model settings
    parser.add_argument('--model', type=str, required=True, help='Model architecture.')
    parser.add_argument('--model_version', type=int, default=0, help='Version of model architecture design.')

    # Dataset settings
    parser.add_argument('--dataset', type=str, required=True, help='Dataset used to train or evaluate model.')

    # Basic NN training settings
    parser.add_argument('--epochs', type=int, default=10, help='Number of training cycles.')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of data examples per training iteration.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Update weights by this factor each iteration.')

    parser.add_argument('--seed', type=int, default=0, help='Random seed for PyTorch library. Used to repeat results of experiment.')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='Method for updating learning rate each epoch.')
    
    args = parser.parse_args()

    if check_arguments(args):
        save_config_file(args)
