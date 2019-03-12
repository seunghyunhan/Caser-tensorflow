import tensorflow as tf
import numpy as np


class Caser(object):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, args):
        super(Caser, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.args = args

        # init args
        self.L = self.args.L
        self.T = self.args.T        
        self.dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.learning_rate = self.args.learning_rate
        self.l2 = self.args.l2
        self.drop_ratio = self.args.drop
        
        # for horizontal conv layer
        self.lengths = [i + 1 for i in range(self.L)]


    def build_model(self):
        """
        """
        self.sequences = tf.placeholder(tf.int32, [None, self.L])
        self.users = tf.placeholder(tf.int32, [None, 1])
        self.items = tf.placeholder(tf.int32, [None, 2*self.T])
        self.is_training = tf.placeholder(tf.bool)
                                             
        # user and item embeddings
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = tf.Variable(initializer([self.num_users, self.dims]))
        self.item_embeddings = tf.Variable(initializer([self.num_items, self.dims]))
        
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = tf.Variable(initializer([self.num_items, self.dims+self.dims]))
        self.b2 = tf.Variable(initializer([self.num_items, 1]))
        
        item_embs = tf.nn.embedding_lookup(self.item_embeddings, self.sequences)
        item_embs = tf.reshape(item_embs, [-1, self.L, self.dims, 1])
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        user_emb = tf.reshape(user_emb, [-1, self.dims])
        
        # vertical convolution layers
        if self.n_v:
            out_v = tf.layers.conv2d(item_embs, 
                                     self.n_v, 
                                     [self.L, 1], 
                                     activation=tf.nn.relu)
            out_v = tf.contrib.layers.flatten(out_v)
            
        # horizontal convolution layers
        out_hs = list()
        if self.n_h:
            for h in self.lengths:
                conv_out = tf.layers.conv2d(item_embs, 
                                            self.n_h, 
                                            [h, self.dims], 
                                            activation=tf.nn.relu)
                conv_out = tf.reshape(conv_out, [-1, self.L-h+1, self.n_h])
                pool_out = tf.layers.max_pooling1d(conv_out, [self.L-h+1], 1)
                pool_out = tf.squeeze(pool_out, 1)
                out_hs.append(pool_out)
            out_h = tf.concat(out_hs, 1)
            
        # concat two convolution layers    
        out = tf.concat([out_v, out_h], 1)
        
        # fully-connected layer
        z = tf.layers.dense(out, self.dims, activation=tf.nn.relu)
        z = tf.layers.dropout(z, self.drop_ratio, self.is_training)
        x = tf.concat([z, user_emb], 1)
        x = tf.reshape(x, [-1, 1, 2*self.dims])
        
        w2 = tf.nn.embedding_lookup(self.W2, self.items)
        b2 = tf.nn.embedding_lookup(self.b2, self.items)
        b2 = tf.squeeze(b2, 2)
        
        # training with negative samples
        pred = tf.squeeze(tf.matmul(x, tf.transpose(w2, perm=[0,2,1])), 1) + b2        
        self.target_pred, negative_pred = tf.split(pred, 2, axis=1)
    
        # loss
        positive_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.target_pred)))
        negative_loss = -tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(negative_pred)))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2

        self.loss = positive_loss + negative_loss + l2_loss
        
        # optimizer
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
        # For test
        self.all_items = tf.placeholder(tf.int32, [None, self.num_items])        
        test_w2 = tf.nn.embedding_lookup(self.W2, self.all_items)
        test_b2 = tf.nn.embedding_lookup(self.b2, self.all_items)        
        test_b2 = tf.reshape(test_b2, [-1, self.num_items])
        self.test_pred = tf.reduce_sum(tf.multiply(x, test_w2), axis=2) + test_b2
    
    
    def train(self, sess, seq_var, user_var, item_var):
        """
        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        """
        loss, _  = sess.run([self.loss, self.train_op], feed_dict={self.sequences: seq_var,
                                                                   self.users: user_var,
                                                                   self.items: item_var,
                                                                   self.is_training: True})
        return loss
    
    
    def predict(self, sess, seq_var, user_var, item_var):
        """
        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        """
        user_var = np.reshape([user_var], [-1, 1])
        item_var = np.reshape(item_var, [-1, self.num_items]) 
           
        pred = sess.run(self.test_pred, feed_dict={self.sequences: seq_var,
                                                   self.users: user_var,
                                                   self.all_items: item_var,
                                                   self.is_training: False})
        pred = np.reshape(pred, [-1])
        return pred