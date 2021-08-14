import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class FENDER(object):
    def __init__(self, data_config, pretrain_data, args):
        self.model_type = 'mf'
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.regs = eval(args.regs)

        self.verbose = args.verbose

        # Placeholder definition
        self.users = tf.placeholder(tf.int32, shape=[None,], name='users')
        self.pos_items = tf.placeholder(tf.int32, shape=[None,], name='pos_items')
        self.neg_items = tf.placeholder(tf.int32, shape=[None,], name='neg_items')

        # Variable definition
        self.weights = self._init_weights()

        # Original embedding.
        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        # Embedding from frequency MF
        f_fre_m = open("../Data/instacart/fre_matrix.pkl","rb")
        self.fre_u_emb = tf.convert_to_tensor(pickle.load(f_fre_m), dtype = tf.float32)
        self.fre_i_emb = tf.convert_to_tensor(pickle.load(f_fre_m), dtype = tf.float32)

        fre_u_e = tf.nn.embedding_lookup(self.fre_u_emb, self.users)
        fre_pos_i_e = tf.nn.embedding_lookup(self.fre_u_emb, self.pos_items)
        fre_neg_i_e = tf.nn.embedding_lookup(self.fre_u_emb, self.neg_items)

        # All predictions for all users.
        self.batch_predictions = self.weights['deconfounder_w']*tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True) + tf.math.subtract(1.0, self.weights['deconfounder_w'])*tf.matmul(fre_u_e, fre_pos_i_e, transpose_a=False, transpose_b=True) 

        # Optimization process.
        self.base_loss, self.reg_loss = self._create_bpr_loss(u_e, pos_i_e, neg_i_e, fre_u_e, fre_pos_i_e, fre_neg_i_e)
        self.loss = self.base_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self._statistics_params()


    def _init_weights(self):
        all_weights = dict()

        initializer = tf.initializers.glorot_normal()

        
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
        all_weights['deconfounder_w'] = tf.Variable(initializer([1,1]), name='deconfounder_w')
        print('using xavier initialization')
        
        return all_weights


    def _create_bpr_loss(self, users, pos_items, neg_items, fre_users, fre_pos_items, fre_neg_items):

        pos_fre = tf.reduce_sum(tf.multiply(fre_users, fre_pos_items), axis=1)
        neg_fre = tf.reduce_sum(tf.multiply(fre_users, fre_neg_items), axis=1)
        pos_scores = self.weights['deconfounder_w']*tf.reduce_sum(tf.multiply(users, pos_items), axis=1) + \
        tf.math.subtract(1.0, self.weights['deconfounder_w'])*pos_fre
        neg_scores = self.weights['deconfounder_w']*tf.reduce_sum(tf.multiply(users, neg_items), axis=1) + \
        tf.math.subtract(1.0, self.weights['deconfounder_w'])*neg_fre
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.regs[0] * regularizer

        return mf_loss, reg_loss


    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.base_loss, self.reg_loss], feed_dict)

    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions