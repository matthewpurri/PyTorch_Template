import argparse

class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # Basic experiment settings
        parser.add_argument('-e', '--epochs', type=int, default=1, 
                            help='Number of epochs to train model (default: 1).')
        parser.add_argument('--model', type=str, default=None, 
                            help='Model architecture name (default: None).')
        parser.add_argument('--dataset', type=str, default=None,
                            help='Dataset to train model (default: None).')
                        
        self.parser = parser

    def parse(self):
        # TODO: Check if model exists
        # TODO: Check if dataset exists
        args = self.parser.parse_args()
        return args
