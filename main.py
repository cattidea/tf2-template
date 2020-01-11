import os

from config_parser.config import CONFIG
from config_parser.argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(CONFIG.name, actions=['train', 'test', 'docs'])
    parser.bind(CONFIG)

    if CONFIG.action == 'train':
        from trainer.train import train
        train()
    elif CONFIG.action == 'test':
        pass
    elif CONFIG.action == 'docs':
        pass
