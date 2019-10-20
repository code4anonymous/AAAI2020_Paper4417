import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import random


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]),'rb'),encoding='latin1'))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def load_social_net_data(data_str1, data_str2, incomplete=False, local_remove=False):
    #read .emd features
    feature_data1 = 'data/'+ data_str1+'.emd'
    feature_data2 = 'data/' + data_str2 + '.emd'

    def read_data(feature_data):
        with open(feature_data,'r') as f:
            lines = f.readlines()
            # num_node, num_feature = lines[0][0], lines[0][1]
            # feature_mat = np.zeros((num_node,num_feature))
            ind=0
            for line in lines:
                content = np.asarray(line.strip().split(), dtype=np.float32)
                if ind==0:
                    num_node, num_feature = int(content[0]), int(content[1])
                    feature_mat = np.zeros((num_node,num_feature))
                    ind+=1
                    continue
                else:
                    feature_mat[int(content[0]),:]=content[1:]
        return feature_mat

    feature_mat1 = read_data(feature_data1)
    feature_mat2 = read_data(feature_data2)



    graph_data1 = 'data/' + data_str1 + '.gpickle'
    graph_data2 = 'data/' + data_str2 + '.gpickle'
    G1 = nx.read_gpickle(graph_data1)
    G2 = nx.read_gpickle(graph_data2)
    idx_G1 = []
    idx_G2 = []
    for n1 in G1.node():
        for n2 in G2.node():
            if G1.node[n1]['old_label'] == G2.node[n2]['old_label']:
                idx_G1.append(n1)
                idx_G2.append(n2)
    print("Total number of identity linkage is " + str(len(idx_G1)))
    idx_G1 = np.asarray(idx_G1, dtype=int)
    idx_G2 = np.asarray(idx_G2, dtype=int)



    adj_data1 = 'data/'+ data_str1+'_adj'
    adj_data2 = 'data/' + data_str2 + '_adj'

    adj1=nx.read_adjlist(adj_data1)
    adj2=nx.read_adjlist(adj_data2)

    if incomplete:         #### 5% of edges are randomly removed
        num_edge_G1 = adj1.number_of_edges()
        num_edge_G2 = adj2.number_of_edges()
        edges_G1 = list(adj1.edges())
        edges_G2 = list(adj2.edges())

        for i in range(int(num_edge_G1*0.03)):
            rand_edge=random.randint(0,num_edge_G1)
            if local_remove:
                while(int(edges_G1[rand_edge][0]) not in idx_G1 and int(edges_G1[rand_edge][1]) not in idx_G1):
                    rand_edge = random.randint(0, num_edge_G1)
                try:
                    adj1.remove_edge(edges_G1[rand_edge][0],edges_G1[rand_edge][1])
                except:
                    print('Already Removed!')
            else:
                while (int(edges_G1[rand_edge][0]) in idx_G1 or int(edges_G1[rand_edge][1]) in idx_G1):
                    rand_edge = random.randint(0, num_edge_G1)
                try:
                    adj1.remove_edge(edges_G1[rand_edge][0], edges_G1[rand_edge][1])
                except:
                    print('Already Removed!')

        for i in range(int(num_edge_G2*0.03)):
            rand_edge=random.randint(0,num_edge_G2)
            if local_remove:
                while(int(edges_G2[rand_edge][0]) not in idx_G2 and int(edges_G2[rand_edge][1]) not in idx_G2):
                    rand_edge = random.randint(0, num_edge_G2)
                try:
                    adj2.remove_edge(edges_G2[rand_edge][0],edges_G2[rand_edge][1])
                except:
                    print('Already Removed!')
            else:
                while (int(edges_G2[rand_edge][0]) in idx_G2 or int(edges_G2[rand_edge][1]) in idx_G2):
                    rand_edge = random.randint(0, num_edge_G2)
                try:
                    adj2.remove_edge(edges_G2[rand_edge][0], edges_G2[rand_edge][1])
                except:
                    print('Already Removed!')



    adj1 = nx.adjacency_matrix(adj1)
    adj2 = nx.adjacency_matrix(adj2)

    feature_mat1 = sp.csr_matrix(feature_mat1).tolil()
    feature_mat2 = sp.csr_matrix(feature_mat2).tolil()



    return adj1, feature_mat1, adj2, feature_mat2, idx_G1, idx_G2

if __name__ == '__main__':
    load_social_net_data('instagram','twitter',incomplete=True)