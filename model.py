from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from gae.SN_layers import GAT, SpGAT
import tensorflow as tf
from tensorflow.contrib.layers import dense_to_sparse

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class GCNModelAE_social(Model):
    def __init__(self, placeholders, num_features1, num_features2, features_nonzero1, features_nonzero2,
                 num_nodes1, num_nodes2, **kwargs):
        super(GCNModelAE_social, self).__init__(**kwargs)

        self.inputs1 = placeholders['features1']
        self.inputs1_gat = placeholders['features1_dense']
        self.input_dim1 = num_features1
        self.features_nonzero1 = features_nonzero1
        self.adj1 = placeholders['adj1']
        self.GID1 = placeholders['GID1']
        self.nodes1 = num_nodes1
        self.bias1 = placeholders['bias1']

        self.inputs2 = placeholders['features2']
        self.inputs2_gat = placeholders['features2_dense']
        self.input_dim2 = num_features2
        self.features_nonzero2 = features_nonzero2
        self.adj2 = placeholders['adj2']
        self.GID2 = placeholders['GID2']
        self.nodes2 = num_nodes2
        self.bias2 = placeholders['bias2']

        # self.train = placeholders['training']

        self.nhead = [8]

        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):

        # Two reconstruction, one from G1 to G1, the other from G2 to G2. Meanwhile, a nonlinear relationship
        # between G1 and G2 embeddings. Using a one layer MLP to predict.

        # For G1, autoencoder
        self.hidden1_G1 = GraphConvolutionSparse(input_dim=self.input_dim1,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj1,
                                              features_nonzero=self.features_nonzero1,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs1)

        self.gat_hid_G1 = SpGAT.inference(inputs=self.inputs1_gat, nb_classes=FLAGS.hidden2,
                                          nb_nodes=self.nodes1, training=True,
                                          attn_drop=self.dropout, ffd_drop=self.dropout, bias_mat=self.bias1,
                                          hid_units=[FLAGS.hidden1], n_heads=self.nhead,
                                          activation=tf.nn.relu, residual=True)

        self.layer1_G1 = tf.concat([self.hidden1_G1,self.gat_hid_G1],axis=-1)

        # self.layer1_G1 = tf.concat([self.gat_hid_G1, self.gat_hid_G1], axis=-1)

        self.hidden2_G1 = GraphConvolution(input_dim=FLAGS.hidden1*2,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj1,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.layer1_G1)

        self.gat_hid_G1 = SpGAT.inference(inputs=tf.expand_dims(self.layer1_G1, axis=0), nb_classes=FLAGS.hidden2,
                                          nb_nodes=self.nodes1, training=True,
                                          attn_drop=self.dropout, ffd_drop=self.dropout, bias_mat=self.bias1,
                                          hid_units=[FLAGS.hidden1], n_heads=self.nhead,
                                          activation=tf.nn.relu, residual=True)

        self.layer2_G1 = tf.concat([self.hidden2_G1, self.gat_hid_G1], axis=-1)

        # self.layer2_G1 = tf.concat([self.gat_hid_G1, self.gat_hid_G1], axis=-1)


        self.hidden3_G1 = GraphConvolution(input_dim=FLAGS.hidden1*2,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj1,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.layer2_G1)

        self.gat_hid_G1 = SpGAT.inference(inputs=tf.expand_dims(self.layer2_G1, axis=0), nb_classes=FLAGS.hidden2,
                                          nb_nodes=self.nodes1, training=True,
                                          attn_drop=self.dropout, ffd_drop=self.dropout, bias_mat=self.bias1,
                                          hid_units=[FLAGS.hidden1], n_heads=self.nhead,
                                          activation=tf.nn.relu, residual=True)

        self.embeddings_G1 = tf.concat([self.hidden3_G1, self.gat_hid_G1], axis=-1)

        # self.embeddings_G1 = tf.concat([self.gat_hid_G1, self.gat_hid_G1], axis=-1)

        ### GAT layers
        # GAT_input1 = tf.sparse_reshape(self.inputs1,shape=[1, self.nodes1, self.input_dim1])
        # GAT_input1 = dense_to_sparse(GAT_input1)

        self.cancat_G1=self.embeddings_G1

        # self.cancat_G1 = tf.concat([self.embeddings_G1,self.gat_hid_G1],axis=-1)
        self.cancat_G1 = tf.layers.dense(tf.nn.relu(self.cancat_G1),FLAGS.hidden2,activation=lambda x: x)


        # self.z_mean = self.embeddings_G1

        self.reconstructions1 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.cancat_G1)

        ##############################################################################

        # For G2, autoencoder, the same network structure as G1.
        self.hidden1_G2 = GraphConvolutionSparse(input_dim=self.input_dim2,
                                                 output_dim=FLAGS.hidden1,
                                                 adj=self.adj2,
                                                 features_nonzero=self.features_nonzero2,
                                                 act=tf.nn.relu,
                                                 dropout=self.dropout,
                                                 logging=self.logging)(self.inputs2)

        self.gat_hid_G2 = SpGAT.inference(inputs=self.inputs2_gat, nb_classes=FLAGS.hidden2, nb_nodes=self.nodes2,
                                          training=True,
                                          attn_drop=self.dropout, ffd_drop=self.dropout, bias_mat=self.bias2,
                                          hid_units=[FLAGS.hidden1],
                                          n_heads=self.nhead,
                                          activation=tf.nn.relu, residual=True)
        #
        self.layer1_G2 = tf.concat([self.hidden1_G2, self.gat_hid_G2], axis=-1)
        # self.layer1_G2 = tf.concat([self.gat_hid_G2, self.gat_hid_G2], axis=-1)

        self.hidden2_G2 = GraphConvolution(input_dim=FLAGS.hidden1*2,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj2,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.layer1_G2)

        self.gat_hid_G2 = SpGAT.inference(inputs=tf.expand_dims(self.layer1_G2, axis=0), nb_classes=FLAGS.hidden2, nb_nodes=self.nodes2,
                                          training=True,
                                          attn_drop=self.dropout, ffd_drop=self.dropout, bias_mat=self.bias2,
                                          hid_units=[FLAGS.hidden1],
                                          n_heads=self.nhead,
                                          activation=tf.nn.relu, residual=True)

        self.layer2_G2 = tf.concat([self.hidden2_G2, self.gat_hid_G2], axis=-1)
        # self.layer2_G2 = tf.concat([self.gat_hid_G2, self.gat_hid_G2], axis=-1)

        self.hidden3_G2 = GraphConvolution(input_dim=FLAGS.hidden1*2,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj2,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.layer2_G2)

        self.gat_hid_G2 = SpGAT.inference(inputs=tf.expand_dims(self.layer2_G2, axis=0), nb_classes=FLAGS.hidden2,
                                          nb_nodes=self.nodes2,
                                          training=True,
                                          attn_drop=self.dropout, ffd_drop=self.dropout, bias_mat=self.bias2,
                                          hid_units=[FLAGS.hidden1],
                                          n_heads=self.nhead,
                                          activation=tf.nn.relu, residual=True)

        self.embeddings_G2 = tf.concat([self.hidden3_G2, self.gat_hid_G2], axis=-1)
        # self.embeddings_G2 = tf.concat([self.gat_hid_G2, self.gat_hid_G2], axis=-1)

        self.cancat_G2=self.embeddings_G2

        # self.cancat_G2 = tf.concat([self.embeddings_G2, self.gat_hid_G2], axis=-1)
        self.cancat_G2 = tf.layers.dense(tf.nn.relu(self.cancat_G2), FLAGS.hidden2, activation=lambda x: x)
        # self.z_mean = self.embeddings_G2

        self.reconstructions2 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                    act=lambda x: x,
                                                    logging=self.logging)(self.cancat_G2)

        #### non-linear mapping from G1 embeddings to G2 embeddings

        self.dense1 = tf.layers.dense(self.embeddings_G1,FLAGS.hidden2,activation=tf.nn.relu)
        self.latentmap = tf.layers.dense(self.dense1, FLAGS.hidden2)


        # classification layers
        # self.match_embedding_G1 = tf.gather(self.cancat_G1, self.GID1)
        self.match_embedding_G1 = tf.gather(self.latentmap, self.GID1)
        self.match_embedding_G2 = tf.gather(self.cancat_G2, self.GID2)

        # self.match_embedding_G1 = tf.gather(tf.sparse_tensor_to_dense(self.inputs1), self.GID1)
        # self.match_embedding_G2 = tf.gather(tf.sparse_tensor_to_dense(self.inputs2), self.GID2)


        self.match_embeddings = tf.concat([self.match_embedding_G1,self.match_embedding_G2],axis=1)



        # self.match_embeddings = tf.concat([tf.sparse_tensor_to_dense(self.inputs1), tf.sparse_tensor_to_dense(self.inputs2)], axis=1)
        # self.match_embeddings = tf.reshape(self.match_embeddings, [-1, FLAGS.hidden2])



        self.match_embeddings = tf.reshape(self.match_embeddings, [-1, FLAGS.hidden2*2])



        self.fcn1 = tf.layers.dense(self.match_embeddings,128,activation=tf.nn.relu)
        # self.fcn2 = tf.layers.dense(self.fcn1, 32, activation=tf.nn.relu)
        self.out = tf.layers.dense(self.fcn1,2)





class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)
