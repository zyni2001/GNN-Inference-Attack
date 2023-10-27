from torch import nn
import torch.nn.functional as F

from lib_gnn_model.diffpool.diffpool_net import DiffPoolNet
import utils.feat_gen_pytorch as feat_gen
from lib_gnn_model.mean_pool.mean_pool_net import MeanPoolNet
from lib_gnn_model.mincut_pool.mincut_pool_net import MinCutPoolNet
from lib_gnn_model.diffpool.diffpool import DiffPool
from lib_gnn_model.mincut_pool.mincut_pool import MinCutPool
from lib_gnn_model.mean_pool.mean_pool import MeanPool

class SubgraphInferNet(nn.Module):
    def __init__(self, feat_dim, embedding_dim, num_classes, max_nodes, feat_gen_method, args):
        super(SubgraphInferNet, self).__init__()

        self.args = args
        self.embedding_dim = embedding_dim
        self.determine_feat_gen_fn(feat_gen_method)
        
        self.graph_pooling = self.determine_graph_pooling_net(feat_dim, num_classes, max_nodes)
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.attack_feat_dim, 128),
        #     nn.Linear(128, 50),
        #     nn.Linear(50, 2)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(self.attack_feat_dim, 128),
            # nn.Dropout(0.25),
            # nn.ReLU(),
            nn.Linear(128, 50),
            # nn.Dropout(0.25),
            # nn.ReLU(),
            nn.Linear(50, 2)
        )



    def determine_graph_pooling_net(self, feat_dim, num_classes, max_nodes):
        if self.args['shadow_model'] == 'diff_pool':
            target_model = DiffPool(feat_dim, num_classes, max_nodes, self.args)
            model = DiffPoolNet(feat_dim, num_classes, max_nodes)
        elif self.args['shadow_model'] == 'mincut_pool':
            target_model = MinCutPool(feat_dim, num_classes, max_nodes, self.args)
            model = MinCutPoolNet(feat_dim, num_classes, max_nodes)
        elif self.args['shadow_model'] == 'mean_pool':
            target_model = MeanPool(feat_dim, num_classes, self.args)
            model = MeanPoolNet(feat_dim, num_classes)
        else:
            raise Exception('unsupported target model')
        
        # target_model.load_model(self.args['target_model_file'])
        # model.load_state_dict(target_model.model.state_dict())
        
        return model
    
    def forward(self, x, adj, mask, graph_embedding):
        self.graph_pooling(x, adj, mask)
        subgraph_embedding = self.graph_pooling.graph_embedding

        joint_embedding = self.feat_gen_fn(graph_embedding, subgraph_embedding).float()
        logit = self.mlp(joint_embedding)
        
        return F.softmax(logit, dim=1)
    
    def loss(self, output, label):
        return F.cross_entropy(output, label)

    def determine_feat_gen_fn(self, feat_gen_method):
        if feat_gen_method == 'concatenate':
            self.feat_gen_fn = feat_gen.concatenate
            self.attack_feat_dim = self.embedding_dim * 2
        elif feat_gen_method == 'cosine_similarity':
            self.feat_gen_fn = feat_gen.cosine_similarity
            self.attack_feat_dim = 1
        elif feat_gen_method == 'l2_distance':
            self.feat_gen_fn = feat_gen.l2_distance
            self.attack_feat_dim = 1
        elif feat_gen_method == 'l1_distance':
            self.feat_gen_fn = feat_gen.l1_distance
            self.attack_feat_dim = 1
        elif feat_gen_method == 'element_l1':
            self.feat_gen_fn = feat_gen.element_l1
            self.attack_feat_dim = self.embedding_dim
        elif feat_gen_method == 'element_l2':
            self.feat_gen_fn = feat_gen.element_l2
            self.attack_feat_dim = self.embedding_dim
        else:
            raise Exception('unsupported feature generation method')
