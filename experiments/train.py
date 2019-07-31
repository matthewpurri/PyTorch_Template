import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from PyTorch_Template.models import get_model
from PyTorch_Template.datasets import get_dataset
from PyTorch_Template.misc import load_config_file, get_optimizer, get_loss_func


class Trainer():
    def __init__(self, args):
        # Load config file
        config_info = load_config_file(args.dataset, args.config_num)

        # Get device type (Use GPU if possible)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        self.dataset = get_dataset(args.dataset)

        # Create dataloaders
        kwargs = {'num_workers': 1,
                  'pin_memory': True}
        self.train_loader = DataLoader(self.dataset, 
                                       batch_size=config_info['batch_size'],
                                       shuffle=True,
                                       **kwargs)
        self.val_loader = DataLoader(self.dataset,
                                     batch_size=config_info['batch_size'],
                                     shuffle=False,
                                     **kwargs)

        # Load model
        self.model = get_model(config_info).to(self.device)

        # Setup optimizer and LR scheduler
        self.optimizer = get_optimizer(config_info)
        # TODO: LR scheduler

        # Get loss function
        self.loss_func = get_loss_func(config_info)


    def train(self):
        train_loss = 0.0

        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, (data, target) in enumerate(tbar):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: {0:4.2f}'.format(train_loss / (i+1)))

    def validate(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Name of dataset to train CNN on.')
    parser.add_argument('--config_num', type=int, help='Number of config file')
    args = parser.parse_args()

    Trainer(args).train()

