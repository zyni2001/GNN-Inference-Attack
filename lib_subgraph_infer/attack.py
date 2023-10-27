import logging
import random

from littleballoffur.exploration_sampling import RandomWalkSampler, SnowBallSampler, ForestFireSampler
import numpy as np
from utils.convert import to_networkx
import torch
import networkx as nx

import utils.feat_gen_numpy as feat_gen


class Attack:
    def __init__(self, target_model, shadow_model, embedding_dim, num_classes, args):
        self.logger = logging.getLogger('attack')

        self.args = args
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.sample_node_ratio = args['sample_node_ratio']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)
        self.target_model.eval()
        self.shadow_model.eval()

    def determine_subsample_cls(self, sample_method, num_nodes=100):
        if sample_method == 'random_walk':
            self.subsample_cls = RandomWalkSampler(number_of_nodes=num_nodes)
        elif sample_method == 'snow_ball':
            self.subsample_cls = SnowBallSampler(number_of_nodes=num_nodes)
        elif sample_method == 'forest_fire':
            self.subsample_cls = ForestFireSampler(number_of_nodes=num_nodes)
        else:
            raise Exception('unsupported sample method')

    def determine_feat_gen_fn(self, feat_gen_method):
        if feat_gen_method == 'concatenate':
            self.feat_gen_fn = feat_gen.concatenate
        elif feat_gen_method == 'cosine_similarity':
            self.feat_gen_fn = feat_gen.cosine_similarity
        elif feat_gen_method == 'l2_distance':
            self.feat_gen_fn = feat_gen.l2_distance
        elif feat_gen_method == 'l1_distance':
            self.feat_gen_fn = feat_gen.l1_distance
        elif feat_gen_method == 'element_l1':
            self.feat_gen_fn = feat_gen.element_l1
        elif feat_gen_method == 'element_l2':
            self.feat_gen_fn = feat_gen.element_l2
        else:
            raise Exception('unsupported feature generation method')
    
    def reindex_graph(self, graph):
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        return nx.relabel_nodes(graph, mapping)

    def prune_nx_graph_using_mask(self, nx_graph, mask):
        """
        Prune the networkx graph based on the given mask. 
        Nodes corresponding to False in the mask will be removed.
        """
        nodes_to_remove = [node for node, m in enumerate(mask) if not m]
        nx_graph.remove_nodes_from(nodes_to_remove)
        return nx_graph

    def generate_subgraph(self, graph):
        # num_sample_nodes = int(graph.num_nodes * self.sample_node_ratio)
        nx_graph = to_networkx(graph, to_undirected=True)

        # # List nodes with no neighbors
        # isolated_nodes = list(nx.isolates(nx_graph))
        # # Remove these nodes from the graph
        # nx_graph.remove_nodes_from(isolated_nodes)
        # nx_graph = self.reindex_graph(nx_graph)
        # num_sample_nodes = int(nx_graph.number_of_nodes() * self.sample_node_ratio)
        
        # Prune the networkx graph based on the mask
        mask_list = graph.mask.tolist()
        nx_graph = self.prune_nx_graph_using_mask(nx_graph, mask_list)
        num_sample_nodes = int(nx_graph.number_of_nodes() * self.sample_node_ratio)

        self.subsample_cls.number_of_nodes = num_sample_nodes
        if not nx.is_connected(nx_graph):
            self.logger.debug('graph unconnected, generate random edge to connect it')
            self.logger.info('graph unconnected, generate random edge to connect it')
            self._connect_nx_graph(nx_graph)
        # if not nx.is_connected(nx_graph):
        subgraph = self.subsample_cls.sample(nx_graph)

        # # 1. Check if all nodes of the subgraph are present in the original graph
        # are_nodes_present = all(node in nx_graph.nodes for node in subgraph.nodes)
        # # 2. Check if all nodes of the subgraph have at least one edge in the original graph
        # are_nodes_connected = all(nx_graph.degree(node) > 0 for node in subgraph.nodes)
        # # Combine the two checks
        # are_all_subgraph_nodes_in_nx_graph = are_nodes_present and are_nodes_connected
        # print('are_all_subgraph_nodes_in_nx_graph', are_all_subgraph_nodes_in_nx_graph)

        # if not nx.is_connected(subgraph):
        # self.logger.info('subgraph unconnected, generate random edge to connect it')
        # self._connect_nx_graph(subgraph)

        return subgraph

    # def generate_subgraph(self, graph):
    #     num_sample_nodes = int(graph.num_nodes * self.sample_node_ratio)
    #     nx_graph = to_networkx(graph, to_undirected=True)
    #     self.subsample_cls.number_of_nodes = num_sample_nodes

    #     if not nx.is_connected(nx_graph):
    #         self.logger.debug('graph unconnected, generate random edge to connect it')
    #         # self._connect_nx_graph(nx_graph)

    #     subgraph = self.subsample_cls.sample(nx_graph)
    #     if not nx.is_connected(subgraph):
    #         self.logger.info('subgraph unconnected, generate random edge to connect it')
    #         # self._connect_nx_graph(subgraph)

    #     return subgraph
    
    def generate_subgraph_data(self, graph, subgraph_nodes):
        subgraph_nodes = list(subgraph_nodes)
        subgraph_x = graph.x[subgraph_nodes]
        x = torch.zeros([self.args['max_nodes'], subgraph_x.shape[1]])
        x[:subgraph_x.shape[0]] = subgraph_x

        # subgraph_adj = torch.tensor(nx.adjacency_matrix(subgraph).todense())
        subgraph_adj = graph.adj[np.ix_(subgraph_nodes, subgraph_nodes)]
        adj = torch.zeros([self.args['max_nodes'], self.args['max_nodes']])
        adj[: subgraph_adj.shape[0], : subgraph_adj.shape[1]] = subgraph_adj

        mask = torch.zeros(self.args['max_nodes'], dtype=torch.bool)
        mask[:subgraph_adj.shape[0]] = 1

        return x, adj, mask

    def generate_input_data(self, graph):
        x = graph.x.reshape([1, graph.x.shape[0], graph.x.shape[1]]).to(self.device)
        adj = graph.adj.reshape([1, graph.adj.shape[0], graph.adj.shape[1]]).to(self.device)
        mask = graph.mask.reshape([1, graph.mask.shape[0]]).to(self.device)

        return x, adj, mask

    def _connect_nx_graph(self, nx_graph):
        components = list(nx.connected_components(nx_graph))
        pre_component = components[0]

        for component in components[1:]:
            v1 = random.choice(tuple(pre_component))
            v2 = random.choice(tuple(component))
            nx_graph.add_edge(v1, v2)
            pre_component = component


if __name__ == '__main__':
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
