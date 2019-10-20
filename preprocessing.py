import numpy as np
import scipy.sparse as sp
import bson
import networkx as nx
import codecs
import json


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj1, features1, features1_dense, adj2, features2, features2_dense, graphID1,
                        graphID2, adj_orig1, adj_orig2, labels,
                        bias1, bias2, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features1']: features1})
    feed_dict.update({placeholders['features1_dense']: features1_dense})
    feed_dict.update({placeholders['features2_dense']: features2_dense})
    feed_dict.update({placeholders['features2']: features2})
    feed_dict.update({placeholders['adj1']: adj1})
    feed_dict.update({placeholders['adj2']: adj2})
    feed_dict.update({placeholders['adj_orig1']: adj_orig1})
    feed_dict.update({placeholders['adj_orig2']: adj_orig2})
    feed_dict.update({placeholders['GID1']: graphID1})
    feed_dict.update({placeholders['GID2']: graphID2})
    feed_dict.update({placeholders['labels']:labels})
    feed_dict.update({placeholders['bias1']: bias1})
    feed_dict.update({placeholders['bias2']: bias2})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(list(all_edge_idx))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def large_social_networks_instagram():
    # count = 0

    degree_thrd = 3


    index = []
    G = nx.Graph()
    with open('/mnt/wzhan139/cross media data/Instagram/instagram_networks.bson', "rb") as f:
        data = bson.decode_file_iter(f, bson.CodecOptions(unicode_decode_error_handler="ignore"))
        count = 0
        for c, d in enumerate(data):
            print("Reading node "+ str(c))
            index.append(d['user_name'])
            G.add_node(d['user_name'])
            count += 1
    with open('/mnt/wzhan139/cross media data/Instagram/instagram_networks.bson', "rb") as f:
        data = bson.decode_file_iter(f, bson.CodecOptions(unicode_decode_error_handler="ignore"))
        for c, d in enumerate(data):
            print("Constructing graph in node " + str(c))
            for i in range(len(d['followers'])):
                if G.has_node(d['followers'][i]['username']):
                    G.add_edge(d['user_name'], d['followers'][i]['username'])
            for j in range(len(d['followees'])):
                if G.has_node(d['followees'][j]['username']):
                    G.add_edge(d['user_name'], d['followees'][j]['username'])

    G2 = nx.convert_node_labels_to_integers(G,label_attribute='old_label')
    num_node = nx.adjacency_matrix(G2).shape[0]
    sparsity = G2.number_of_edges() / num_node ** 2
    print("no thredshold graph sparsity is " + str(sparsity))
    print(nx.info(G2))
    nx.write_gpickle(G2, "instagram.nothred.gpickle")

    remove_node=[]
    for n, d in G2.nodes(data=True):
        if G2.degree(n)<degree_thrd:
            remove_node.append(n)
    G2.remove_nodes_from(np.asarray(remove_node))
    G2 = nx.convert_node_labels_to_integers(G2)

    num_node=nx.adjacency_matrix(G2).shape[0]
    G3 = nx.from_scipy_sparse_matrix(sp.dia_matrix((np.ones(num_node), 0), shape=nx.adjacency_matrix(G2).shape))
    G4=nx.compose(G2,G3)
    nx.write_gpickle(G4, "instagram.gpickle")
    nx.write_adjlist(G4, "instagram_adj")
    nx.write_edgelist(G4, "instagram_edgelist")

    sparsity = G4.number_of_edges()/num_node**2
    print("sparsity is "+ str(sparsity))
    print(nx.info(G4))

    ###### run Node2Vec   python src/main.py --input "/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/gae/gae/instagram_edgelist"
    # --output emb/instagram.emd --dimensions=64


# def load_social_net_data(data_path):
#     #read .emd features
#     feature_data = data_path+'.emd'
#     with open(feature_data,'r') as f:
#         lines = f.readlines()
#         # num_node, num_feature = lines[0][0], lines[0][1]
#         # feature_mat = np.zeros((num_node,num_feature))
#         ind=0
#         for line in lines:
#             content = np.asarray(line.strip().split(), dtype=np.float32)
#             if ind==0:
#                 num_node, num_feature = int(content[0]), int(content[1])
#                 feature_mat = np.zeros((num_node,num_feature))
#                 ind+=1
#                 continue
#             else:
#                 feature_mat[int(content[0]),:]=content[1:]
#
#     adj_data = data_path+'_adj'
#     adj = nx.read_adjlist(adj_data)
#
#     return adj, feature_mat


def large_social_networks_twitter():
    # count = 0

    degree_thrd = 3


    index = []
    G = nx.Graph()
    with open('/mnt/wzhan139/cross media data/Twitter/twitter_followees.bson', "rb") as f:
        data = bson.decode_file_iter(f, bson.CodecOptions(unicode_decode_error_handler="ignore"))
        count = 0
        for c, d in enumerate(data):
            print("Reading node "+ str(c))
            index.append(d['user_name'])
            G.add_node(d['user_name'])
            count += 1
    with open('/mnt/wzhan139/cross media data/Twitter/twitter_followees.bson', "rb") as f:
        data = bson.decode_file_iter(f, bson.CodecOptions(unicode_decode_error_handler="ignore"))
        for c, d in enumerate(data):
            print("Constructing graph in node " + str(c))
            for j in range(len(d['followees'])):
                if G.has_node(d['followees'][j]['screen_name']):
                    G.add_edge(d['user_name'], d['followees'][j]['screen_name'])
    with open('/mnt/wzhan139/cross media data/Twitter/twitter_followers.bson', "rb") as f:
        data = bson.decode_file_iter(f, bson.CodecOptions(unicode_decode_error_handler="ignore"))
        for c, d in enumerate(data):
            print("Constructing graph in node " + str(c))
            for i in range(len(d['followers'])):
                if G.has_node(d['followers'][i]['screen_name']):
                    G.add_edge(d['user_name'], d['followers'][i]['screen_name'])


    G2 = nx.convert_node_labels_to_integers(G,label_attribute='old_label')
    num_node = nx.adjacency_matrix(G2).shape[0]
    sparsity = G2.number_of_edges() / num_node ** 2
    print("no thredshold graph sparsity is " + str(sparsity))
    print(nx.info(G2))
    nx.write_gpickle(G2, "twitter.nothred.gpickle")

    remove_node=[]
    for n, d in G2.nodes(data=True):
        if G2.degree(n)<degree_thrd:
            remove_node.append(n)
    G2.remove_nodes_from(np.asarray(remove_node))
    G2 = nx.convert_node_labels_to_integers(G2)

    num_node=nx.adjacency_matrix(G2).shape[0]
    G3 = nx.from_scipy_sparse_matrix(sp.dia_matrix((np.ones(num_node), 0), shape=nx.adjacency_matrix(G2).shape))
    G4=nx.compose(G2,G3)
    nx.write_gpickle(G4, "twitter.gpickle")
    nx.write_adjlist(G4, "twitter_adj")
    nx.write_edgelist(G4, "twitter_edgelist")

    sparsity = G4.number_of_edges()/num_node**2
    print("sparsity is "+ str(sparsity))
    print(nx.info(G4))


def large_social_networks_flickr():
    count = 0

    degree_thrd = 3


    index = []
    G = nx.Graph()
    with codecs.open('/mnt/wzhan139/cross media data/Flickr/flickr_friends.json','rU','utf-8') as f:
        for line in f:
            try:
                data=json.loads(line)
                print("Reading node "+str(count))
                index.append(data['user_name'])
                G.add_node(data['user_name'])
                count+=1
            except:
                break

    count = 0
    with codecs.open('/mnt/wzhan139/cross media data/Flickr/flickr_friends.json','rU','utf-8') as f:
        for line in f:
            try:
                data=json.loads(line)
                print("Constructing graph in node "+str(count))
                for j in range(len(data['following'])):
                    if G.has_node(data['following'][j]['username']):
                        G.add_edge(data['user_name'], data['following'][j]['username'])
            except:
                break

    G2 = nx.convert_node_labels_to_integers(G,label_attribute='old_label')
    num_node = nx.adjacency_matrix(G2).shape[0]
    sparsity = G2.number_of_edges() / num_node ** 2
    print("no thredshold graph sparsity is " + str(sparsity))
    print(nx.info(G2))
    nx.write_gpickle(G2, "flickr.nothred.gpickle")

    remove_node=[]
    for n, d in G2.nodes(data=True):
        if G2.degree(n)<degree_thrd:
            remove_node.append(n)
    G2.remove_nodes_from(np.asarray(remove_node))
    G2 = nx.convert_node_labels_to_integers(G2)

    num_node=nx.adjacency_matrix(G2).shape[0]
    G3 = nx.from_scipy_sparse_matrix(sp.dia_matrix((np.ones(num_node), 0), shape=nx.adjacency_matrix(G2).shape))
    G4=nx.compose(G2,G3)
    nx.write_gpickle(G4, "flickr.gpickle")
    nx.write_adjlist(G4, "flickr_adj")
    nx.write_edgelist(G4, "flickr_edgelist")

    sparsity = G4.number_of_edges()/num_node**2
    print("sparsity is "+ str(sparsity))
    print(nx.info(G4))


def rebuttal_data():
    for dataname in ['flickr','lastfm']:

        count = 0

        degree_thrd = 3

        index = []
        G = nx.Graph()
        with open('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/gae/'+dataname+'/'+dataname+'.nodes') as f:
            for line in f:
                try:
                    word = line.split()
                    if(count%1000==0): print("Processed node " + word[0])
                    index.append(word[0])
                    G.add_node(word[0], old_label=word[1])
                    count += 1
                except:
                    break
        print("Reading node Done!")

        count = 0
        with open('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/gae/'+dataname+'/'+dataname+'.edges') as f:
            for line in f:
                try:
                    link = line.split()
                    if (count % 100000==0): print("Processed edge " + link[0])
                    # print("Constructing graph in node " + str(count))
                    G.add_edge(link[0], link[1])
                    count += 1
                except:
                    break
        print("Reading edges Done!")

        # G2 = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        num_node = nx.adjacency_matrix(G).shape[0]
        sparsity = G.number_of_edges() / num_node ** 2
        print("no thredshold graph sparsity is " + str(sparsity))
        print(nx.info(G))
        nx.write_gpickle(G, "large_"+dataname+".nothred.gpickle")

        remove_node = []
        for n, d in G.nodes(data=True):
            if G.degree(n) < degree_thrd:
                remove_node.append(n)
        G.remove_nodes_from(np.asarray(remove_node))
        G = nx.convert_node_labels_to_integers(G)

        num_node = nx.adjacency_matrix(G).shape[0]
        G3 = nx.from_scipy_sparse_matrix(sp.dia_matrix((np.ones(num_node), 0), shape=nx.adjacency_matrix(G).shape))
        G4 = nx.compose(G, G3)
        nx.write_gpickle(G4, "large_"+dataname+".gpickle")
        nx.write_adjlist(G4, "large_"+dataname+"_adj")
        nx.write_edgelist(G4, "large_"+dataname+"_edgelist")

        sparsity = G4.number_of_edges() / num_node ** 2
        print("sparsity is " + str(sparsity))
        print(nx.info(G4))


if __name__ == '__main__':
    rebuttal_data()