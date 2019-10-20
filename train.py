from __future__ import division
from __future__ import print_function

import time
import os


import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.utils import shuffle
import sklearn.preprocessing as sklp
import matplotlib.pyplot as plt


from gae.optimizer import OptimizerAE, OptimizerVAE, OptimizerAE_social
from gae.input_data import load_data, load_social_net_data
from gae.model import GCNModelAE, GCNModelVAE, GCNModelAE_social
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from gae.SN_layers import preprocess_adj_bias

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 80, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 3.')

flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('patience', 4, 'Early stopping patience.')
flags.DEFINE_float('mini_delta', 0.0005, 'Early stopping step difference.')
# flags.DEFINE_float('alpha', 1, 'Weight for 2nd graph reconstruction.')
# flags.DEFINE_float('beta', 1, 'Weight for classification loss.')

flags.DEFINE_string('model', 'gcn_ae_social', 'Model string.')
flags.DEFINE_string('model_save', 'performance/checkpoint', 'Model Checkpoint.')
flags.DEFINE_string('dataset1', 'instagram', 'Dataset string.')
flags.DEFINE_string('dataset2', 'twitter', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
# flags.DEFINE_integer('training', 1, 'Is training or not')

model_str = FLAGS.model
# dataset_str = FLAGS.dataset

# Load data
# adj, features = load_data(dataset_str)

os.environ["CUDA_VISIBLE_DEVICES"]="1"

adj1, features1_raw, adj2, features2_raw, graphID1, graphID2 = load_social_net_data(FLAGS.dataset1, FLAGS.dataset2, incomplete=False, local_remove=False)


    # Negative Sampling
def neg_sampling(IDlength, GID):
    sampleN = np.int(GID[0].shape[0]/3)
    Res_GID1 = np.setdiff1d(range(IDlength), GID[0])
    Res_GID2 = np.setdiff1d(range(IDlength), GID[1])
    train_GID1 = np.concatenate((GID[0], shuffle(Res_GID1)[0:sampleN], shuffle(GID[0])[0:sampleN], shuffle(Res_GID1)[0:sampleN]))
    train_GID2 = np.concatenate((GID[1], shuffle(GID[1])[0:sampleN], shuffle(Res_GID2)[0:sampleN], shuffle(Res_GID2)[0:sampleN]))
    label = np.concatenate((np.ones(GID[0].shape[0]),np.zeros(sampleN*3)))
    return [train_GID1, train_GID2], label





# Store original adjacency matrix (without diagonal entries) for later
def neg_sampling_weight(adj):
    dense = adj.count_nonzero()/adj1.shape[0]**2
    tmp = sp.rand(adj.shape[0], adj.shape[1], density=dense*3,format="csr",random_state=40)
    adj += (tmp*2).astype(np.int)
    return sklp.binarize(adj,1)


adj_orig1 = neg_sampling_weight(adj1)
adj_orig1 = adj_orig1 - sp.dia_matrix((adj_orig1.diagonal()[np.newaxis, :], [0]), shape=adj_orig1.shape)
adj_orig1.eliminate_zeros()
biases1 = preprocess_adj_bias(adj1)
adj_orig1 = sparse_to_tuple(adj_orig1.tocoo())

adj_orig2 = neg_sampling_weight(adj2)
adj_orig2 = adj_orig2 - sp.dia_matrix((adj_orig2.diagonal()[np.newaxis, :], [0]), shape=adj_orig2.shape)
adj_orig2.eliminate_zeros()
biases2 = preprocess_adj_bias(adj2)
adj_orig2 = sparse_to_tuple(adj_orig2.tocoo())

# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
# adj = adj_train





if FLAGS.features == 0:
    features1_raw = sp.identity(features1_raw.shape[0])  # featureless

# Some preprocessing
adj_norm1 = preprocess_graph(adj1)
adj_norm2 = preprocess_graph(adj2)

num_nodes1 = adj1.shape[0]
num_nodes2 = adj2.shape[0]

features1_dense = features1_raw.todense()
features1_dense = features1_dense[np.newaxis]
features1 = sparse_to_tuple(features1_raw.tocoo())
num_features1 = features1[2][1]
features_nonzero1 = features1[1].shape[0]

features2_dense = features2_raw.todense()
features2_dense = features2_dense[np.newaxis]
features2 = sparse_to_tuple(features2_raw.tocoo())
num_features2 = features2[2][1]
features_nonzero2 = features2[1].shape[0]




grid_search_alpha = [0, 0.01, 0.1, 1, 10, 100]
grid_search_beta = [0, 0.01, 0.1, 1, 10, 100]
# grid_search_alpha = [10]
# grid_search_beta = [1]
retest_n = 10

grid_search_acc_mat = np.zeros([len(grid_search_alpha),len(grid_search_beta)],dtype=np.float32)
grid_search_f1_mat = np.zeros([len(grid_search_alpha),len(grid_search_beta)],dtype=np.float32)
grid_search_precision_mat = np.zeros([len(grid_search_alpha),len(grid_search_beta)],dtype=np.float32)
grid_search_recall_mat = np.zeros([len(grid_search_alpha),len(grid_search_beta)],dtype=np.float32)

grid_search_hist_loss = []
grid_search_hist_match_loss = []
grid_search_hist_acc = []

# fig, ax = plt.subplots(nrows=len(grid_search_alpha),ncols=len(grid_search_beta))


for ind_alpha, subalpha in enumerate(grid_search_alpha):
    for ind_beta, subbeta in enumerate(grid_search_beta):

        test_acc_average = 0
        test_f1_average = 0

        for rep in range(retest_n):

            # Define placeholders
            placeholders = {
                'features1': tf.sparse_placeholder(tf.float32),
                'features1_dense': tf.placeholder(tf.float32, shape=(1, num_nodes1, num_features1)),
                'features2': tf.sparse_placeholder(tf.float32),
                'features2_dense': tf.placeholder(tf.float32, shape=(1, num_nodes2, num_features2)),
                'adj1': tf.sparse_placeholder(tf.float32),
                'adj_orig1': tf.sparse_placeholder(tf.float32),
                'adj2': tf.sparse_placeholder(tf.float32),
                'adj_orig2': tf.sparse_placeholder(tf.float32),
                'dropout': tf.placeholder_with_default(0., shape=()),
                'GID1': tf.placeholder(tf.int32),
                'GID2': tf.placeholder(tf.int32),
                'labels': tf.placeholder(tf.int32),
                'bias1': tf.sparse_placeholder(dtype=tf.float32),
                'bias2': tf.sparse_placeholder(dtype=tf.float32)
                # 'training': tf.placeholder(tf.int64)
            }

            # Create model
            model = None
            if model_str == 'gcn_ae':
                model = GCNModelAE(placeholders, num_features1, features_nonzero1)
            elif model_str == 'gcn_vae':
                model = GCNModelVAE(placeholders, num_features1, num_nodes1, features_nonzero1)
            elif model_str == 'gcn_ae_social':
                model = GCNModelAE_social(placeholders, num_features1, num_features2, features_nonzero1,
                                          features_nonzero2,
                                          num_nodes1, num_nodes2)

            pos_weight1 = float(adj1.shape[0] * adj1.shape[0] - adj1.sum()) / adj1.sum()
            norm1 = adj1.shape[0] * adj1.shape[0] / float((adj1.shape[0] * adj1.shape[0] - adj1.sum()) * 2)

            pos_weight2 = float(adj2.shape[0] * adj2.shape[0] - adj2.sum()) / adj2.sum()
            norm2 = adj2.shape[0] * adj2.shape[0] / float((adj2.shape[0] * adj2.shape[0] - adj2.sum()) * 2)

            sampling_train_all, sampling_train_all_labels = neg_sampling(adj1.shape[0], [graphID1, graphID2])


            sampling_train_all[0], sampling_train_all[1], sampling_train_all_labels = shuffle(sampling_train_all[0],
                                                                                              sampling_train_all[1],
                                                                                              sampling_train_all_labels)

            ID_len = sampling_train_all[0].shape[0]

            sampling_train_GID, sampling_train_labels = [sampling_train_all[0][:int(ID_len * 0.6)],
                                                         sampling_train_all[1][:int(ID_len * 0.6)]], \
                                                        sampling_train_all_labels[:int(ID_len * 0.6)]

            sampling_val_GID, sampling_val_labels = [sampling_train_all[0][int(ID_len * 0.6):int(ID_len * 0.8)],
                                                     sampling_train_all[1][int(ID_len * 0.6):int(ID_len * 0.8)]], \
                                                    sampling_train_all_labels[int(ID_len * 0.6):int(ID_len * 0.8)]

            sampling_test_GID, sampling_test_labels = [sampling_train_all[0][int(ID_len * 0.8):],
                                                       sampling_train_all[1][int(
                                                           ID_len * 0.8):]], sampling_train_all_labels[
                                                                             int(ID_len * 0.8):]


        # Optimizer
            with tf.name_scope('optimizer'):
                if model_str == 'gcn_ae':
                    opt = OptimizerAE(preds=model.reconstructions,
                                      labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                  validate_indices=False), [-1]),
                                      pos_weight=pos_weight1,
                                      norm=norm1)
                elif model_str == 'gcn_vae':
                    opt = OptimizerVAE(preds=model.reconstructions,
                                       labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                   validate_indices=False), [-1]),
                                       model=model, num_nodes=num_nodes1,
                                       pos_weight=pos_weight1,
                                       norm=norm1)
                elif model_str == 'gcn_ae_social':
                    opt = OptimizerAE_social(preds1=model.reconstructions1,
                                            recons1=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj1'],
                                                                                      validate_indices=False), [-1]),
                                            preds2=model.reconstructions2,
                                            recons2=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj2'],
                                                                                       validate_indices=False), [-1]),
                                            pos_weight1=pos_weight1,norm1=norm1,
                                            pos_weight2=pos_weight2,norm2=norm2,
                                            embedding1=model.cancat_G1,
                                            embedding2=model.cancat_G2,
                                            predict = model.out,
                                            labels = placeholders['labels'],
                                            matchID1=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig1'],
                                                                                      validate_indices=False), [-1]),
                                            matchID2=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig2'],
                                                                                      validate_indices=False), [-1]),
                                            beta = subbeta, alpha = subalpha,

                                            )




            # Initialize session
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            cost_val = []
            acc_val = []




            # cost_val = []
            # acc_val = []
            val_roc_score = []

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('performance/loss', sess.graph)

            saver = tf.train.Saver()

            vl_acc_mx = 0.0
            vl_loss_mn = np.inf

            # adj_label = adj_orig1 + sp.eye(adj1.shape[0])
            # adj_label = sparse_to_tuple(adj_label)

            # Train model
            hist_train_loss = []
            hist_train_match_loss = []
            hist_train_acc = []
            hist_val_loss = []
            hist_val_match_loss = []
            hist_val_acc = []
            patience_cnt = 0
            for epoch in range(FLAGS.epochs):

                t = time.time()
                # Construct feed dictionary
                feed_dict = construct_feed_dict(adj_norm1, features1, features1_dense, adj_norm2, features2, features2_dense,
                                                sampling_train_GID[0], sampling_train_GID[1],
                                                adj_orig1, adj_orig2, sampling_train_labels, biases1, biases2, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                # feed_dict.update({placeholders['training']:FLAGS.training})
                # Run single weight update
                outs = sess.run([opt.opt_op, opt.cost, opt.match, opt.acc, summary_op, opt.breakp], feed_dict=feed_dict)

                train_avg_cost = outs[1]
                hist_train_loss.append(train_avg_cost)
                train_loss_match = outs[2]
                hist_train_match_loss.append(train_loss_match)
                train_acc = outs[3]
                hist_train_acc.append(train_acc)

                summary_writer.add_summary(outs[4],epoch)

                print('breakp is '+ str(outs[5]))


                # validation
                vl_step = 0

                feed_dict = construct_feed_dict(adj_norm1, features1, features1_dense, adj_norm2, features2, features2_dense,
                                                sampling_val_GID[0], sampling_val_GID[1],
                                                adj_orig1, adj_orig2, sampling_val_labels, biases1, biases2, placeholders)
                feed_dict.update({placeholders['dropout']: 0})
                # feed_dict.update({placeholders['training']: 0})
                outs = sess.run([opt.cost, opt.match, opt.acc], feed_dict=feed_dict)
                val_avg_cost = outs[0]
                hist_val_loss.append(val_avg_cost)
                val_loss_match = outs[1]
                hist_val_match_loss.append(val_loss_match)
                val_acc = outs[2]
                hist_val_acc.append(val_acc)

                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_cost),
                      "train_match_loss={: .5f}".format(train_loss_match),
                      "train_acc=", "{:.5f}".format(train_acc),
                      "|| val_loss={:.5f}".format(val_avg_cost),
                      "val_match_loss={: .5f}".format(val_loss_match),
                      "val_acc=", "{:.5f}".format(val_acc),
                      "time=", "{:.5f}".format(time.time() - t))

                if epoch>15 and hist_val_loss[epoch-1]-hist_val_loss[epoch]>FLAGS.mini_delta:
                # if epoch > 25 and hist_val_acc[epoch] > 0.50:
                    patience_cnt+=1
                else:
                    patience_cnt=0

                if patience_cnt>FLAGS.patience or epoch >= FLAGS.epochs-1:



                    ##### testing
                    feed_dict = construct_feed_dict(adj_norm1, features1, features1_dense, adj_norm2, features2, features2_dense,
                                                    sampling_test_GID[0], sampling_test_GID[1], adj_orig1, adj_orig2, sampling_test_labels,
                                                    biases1, biases2, placeholders)
                    feed_dict.update({placeholders['dropout']: 0})
                    outs = sess.run([opt.cost, opt.match, opt.acc, opt.f1], feed_dict=feed_dict)
                    test_cost = outs[0]
                    test_loss_match = outs[1]
                    test_acc = outs[2]
                    test_f1 = outs[3][0]
                    test_precision = outs[3][1]
                    test_recall = outs[3][2]

                    test_acc_average+=test_acc
                    test_f1_average+=test_f1

                    tp_view = outs[3][3]
                    fp_view = outs[3][4]
                    fn_view = outs[3][5]
                    tn_view = outs[3][6]

                    print("test_tp={},".format(tp_view),
                          "test_fp={},".format(fp_view),
                          "test_fn={},".format(fn_view),
                          "test_tn={},".format(tn_view)
                          )




                    print("Early stopping ......")
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_cost),
                          "train_match_loss={: .5f}".format(train_loss_match),
                          "train_acc=", "{:.5f}".format(train_acc),
                          "|| test_loss={:.5f}".format(test_cost),
                          "test_match_loss={: .5f}".format(test_loss_match),
                          "test_acc=", "{:.5f}".format(test_acc),
                          "test_f1=", "{:.5f}".format(test_f1),
                          "test_precision=", "{:.5f}".format(test_precision),
                          "test_recall=", "{:.5f}".format(test_recall),
                          "time=", "{:.5f}".format(time.time() - t))

                    break




            # grid_search_hist_loss.append(hist_train_loss)
            # grid_search_hist_loss.append(hist_val_loss)
            # grid_search_hist_match_loss.append(hist_train_match_loss)
            # grid_search_hist_match_loss.append(hist_val_match_loss)
            # grid_search_hist_acc.append(hist_train_acc)
            # grid_search_hist_acc.append(hist_val_acc)


            sess.close()
            tf.keras.backend.clear_session()

        grid_search_acc_mat[ind_alpha][ind_beta] = test_acc_average / retest_n
        grid_search_f1_mat[ind_alpha][ind_beta] = test_f1_average / retest_n


def save_result_fig(row, col, xlist, filename, yrange=None):
    fig, ax = plt.subplots(nrows=len(row),ncols=len(col),squeeze=False)
    fig.tight_layout()
    for i in range(len(row)):
        for j in range(len(col)):
            g1y=np.array(xlist[2*(i*len(col)+j)])
            g1x=range(len(g1y))
            ax[i][j].plot(g1x,g1y, color = 'blue')
            g2y = np.array(xlist[2*(i*len(col)+j)+1])
            g2x = range(len(g2y))
            ax[i][j].plot(g2x, g2y, color='red')
            ax[i][j].set_title(r'$\alpha$ = ' + str(row[i]) + r' and $\beta$ = ' + str(col[j]), {'fontsize':6})
            if yrange is not None:
                ax[i][j].set_ylim(yrange[0],yrange[1])
    plt.savefig(filename, dpi=300)


# save_result_fig(grid_search_alpha, grid_search_beta, grid_search_hist_loss, 'performance/grid_search_loss.jpg')
# save_result_fig(grid_search_alpha, grid_search_beta, grid_search_hist_match_loss, 'performance/grid_search_match_loss.jpg', [0.5, 0.8])
# save_result_fig(grid_search_alpha, grid_search_beta, grid_search_hist_acc, 'performance/grid_search_acc.jpg', [0.41, 0.8])


# print("Optimization Finished!")
#
# # roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))


# print("grid search for acc, result matrix is :")
# print(grid_search_acc_mat)
# np.save('acc_mat',grid_search_acc_mat)
# print("grid search for f1, result matrix is :")
# print(grid_search_f1_mat)
# np.save('f1_mat',grid_search_f1_mat)