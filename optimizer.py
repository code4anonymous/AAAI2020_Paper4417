import tensorflow as tf
# from sklearn.metrics.pairwise import cosine_similarity

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class OptimizerAE_social(object):
    def __init__(self, preds1, recons1, preds2, recons2, pos_weight1, norm1, pos_weight2, norm2,
                 embedding1, embedding2, predict, labels, matchID1, matchID2, alpha=1, beta=1):
        preds_sub1 = preds1
        labels_sub1 = recons1
        preds_sub2 = preds2
        labels_sub2 = recons2

        self.labels = tf.one_hot(labels, 2)
        #
        # def nag_sample_weight(labels):
        #     labels_weight = tf.cast(labels,dtype=tf.int32) + tf.random_uniform(shape=tf.shape(labels),minval=0,maxval=2,dtype=tf.int32)
        #     labels_weight = tf.one_hot(labels_weight,2)
        #     labels_weight = tf.where(tf.equal(labels_weight,1))
        #     return labels_weight[:,1]
        #
        # weight_sub1=nag_sample_weight(labels_sub1)
        # weight_sub2=nag_sample_weight(labels_sub2)

        self.cost1 = tf.reduce_mean(
            tf.losses.mean_squared_error(predictions=preds_sub1, labels=labels_sub1, weights= matchID1))
        self.cost2 = tf.reduce_mean(
            tf.losses.mean_squared_error(predictions=preds_sub2, labels=labels_sub2, weights=matchID2))

        ######## 1st order local properties
        def pairwise_dist(A,B):
            with tf.variable_scope('pairwise_dist'):
                # squared norms of each row in A and B
                na = tf.reduce_sum(tf.square(A), 1)
                nb = tf.reduce_sum(tf.square(B), 1)

                # na as a row and nb as a co"lumn vectors
                na = tf.reshape(na, [-1, 1])
                nb = tf.reshape(nb, [1, -1])

                # return pairwise euclidead difference matrix
                D = tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0)
            return D



        self.cost1_local = tf.reduce_sum(tf.multiply(tf.reshape(pairwise_dist(embedding1,embedding1),[-1]),labels_sub1))
        self.cost2_local = tf.reduce_sum(tf.multiply(tf.reshape(pairwise_dist(embedding2, embedding2),[-1]), labels_sub2))
        self.cost_local = self.cost1_local + self.cost2_local

        self.breakp = self.cost_local

        # subgraph1 = tf.gather(embedding1, matchID1)
        # subgraph1 = tf.nn.l2_normalize(embedding1,1)
        # subgraph2 = tf.gather(embedding2, matchID2)
        # subgraph2 = tf.nn.l2_normalize(embedding2, 1)
        # matchdiff = tf.losses.cosine_distance(subgraph1,subgraph2,axis=1)
        matchdiff = tf.reduce_sum(tf.reshape(pairwise_dist(embedding2, embedding2),[-1]))
        # self.match = tf.reduce_mean(matchdiff-1)

        self.match = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=self.labels))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.cost = alpha*self.cost1 + alpha*self.cost2 + self.match + beta*self.cost_local+matchdiff*0.1
        # self.cost=alpha*self.cost1 + alpha*self.cost2
        self.opt_op = self.optimizer.minimize(self.cost)
        # self.grads_vars = self.optimizer.compute_gradients(self.cost)



        self.acc = self.classification_accuracy(predict, self.labels)
        self.f1 = self.micro_f1(predict, self.labels)
        # self.tarIDs = matchID2

        tf.summary.scalar('Reconstruction loss graph 1', self.cost1)
        tf.summary.scalar('Reconstruction loss graph 2', self.cost2)
        tf.summary.scalar('Match loss',  self.match)

        # self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
        #                                    tf.cast(labels_sub, tf.int32))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def hit_accuracy(self, embedding1, embedding2, matchID1):
        subgraph1 = tf.gather(embedding1,matchID1)
        normalize_subgraph1 = tf.nn.l2_normalize(subgraph1,1)
        normalize_embedding2 = tf.nn.l2_normalize(embedding2,1)
        cos_similar = tf.matmul(normalize_subgraph1,tf.transpose(normalize_embedding2))
        _, ind = tf.nn.top_k(cos_similar, 5)
        return ind

    def classification_accuracy(self, predict, labels):
        correct_prediction = tf.equal(tf.argmax(predict,1),tf.argmax(labels,1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(accuracy_all)

    def micro_f1(self, logits, labels):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])


        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1))
        fp = tf.count_nonzero(predicted * (labels - 1))
        fn = tf.count_nonzero((predicted - 1) * labels)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure, precision, recall, tp, fp, fn, tn
