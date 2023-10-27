import logging
import pickle
import os
import torch


class GNNBase:
    def __init__(self, args):
        self.logger = logging.getLogger('gnn')

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.embedding_dim = 0

    def save_model(self, save_path):
        self.logger.info('saving model')
        temp = self.model.cpu().state_dict()
        # if save_path directory does not exist, create it
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(temp, save_path)

    def load_model(self, save_path):
        self.logger.info('loading model')
        self.logger.info("Current device: %s" % (torch.cuda.current_device()))
        temp = torch.load(save_path)
        self.model.load_state_dict(temp)

    def save_paras(self, save_path):
        self.logger.info('saving paras')
        self.paras = {
            'embedding_dim': self.embedding_dim
        }
        pickle.dump(self.paras, open(save_path, 'wb'))

    def load_paras(self, save_path):
        self.logger.info('loading paras')
        return pickle.load(open(save_path, 'rb'))
