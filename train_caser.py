import argparse
from time import time

from caser import Caser
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import *


class Recommender(object):
    """
    Contains attributes and methods that needed to train a sequential
    recommendation model. Models are trained by many tuples of
    (users, sequences, targets, negatives) and negatives are from negative
    sampling: for any known tuple of (user, sequence, targets), one or more
    items are randomly sampled to act as negatives.


    Parameters
    ----------

    args: args,
        Model-related arguments, like latent dimensions.
    """
    def __init__(self, args=None):
        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.args = args

        # learning related
        self._batch_size = self.args.batch_size
        self._n_iter = self.args.n_iter
        self._neg_samples = self.args.neg_samples

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions, L, T):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.args)

        self._net.build_model(L, T)
        
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init) 
        

    def fit(self, train, test, verbose=False):
        """
        The general training loop to fit the model

        Parameters
        ----------

        train: :class:`spotlight.interactions.Interactions`
            training instances, also contains test sequences
        test: :class:`spotlight.interactions.Interactions`
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """

        # convert to sequences, targets and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        L, T = train.sequences.L, train.sequences.T

        n_train = sequences_np.shape[0]

        output_str = 'total training instances: %d' % n_train
        print(output_str)

        if not self._initialized:
            self._initialize(train, L, T)

            
        start_epoch = 0
        for epoch_num in range(start_epoch, self._n_iter):

            t1 = time()
            users_np, sequences_np, targets_np = shuffle(users_np,
                                                         sequences_np,
                                                         targets_np)

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            epoch_loss = 0.0
            for (minibatch_num,
                 (batch_users,
                  batch_sequences,
                  batch_targets,
                  batch_negatives)) in enumerate(minibatch(users_np,
                                                           sequences_np,
                                                           targets_np,
                                                           negatives_np,
                                             batch_size=self._batch_size)):
                
                
                items_to_predict = np.concatenate((batch_targets, batch_negatives), 1)
                loss = self._net.train(self.sess, 
                                             batch_sequences,
                                             batch_users,
                                             items_to_predict)
                epoch_loss += loss
            epoch_loss /= minibatch_num + 1

            t2 = time()
            if verbose and (epoch_num + 1) % 10 == 0:
                precision, recall, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])
                output_str = "Epoch %d [%.1f s]\tloss=%.4f, map=%.4f, " \
                             "prec@1=%.4f, prec@5=%.4f, prec@10=%.4f, " \
                             "recall@1=%.4f, recall@5=%.4f, recall@10=%.4f, [%.1f s]" % (epoch_num + 1,
                                                                                         t2 - t1,
                                                                                         epoch_loss,
                                                                                         mean_aps,
                                                                                         np.mean(precision[0]),
                                                                                         np.mean(precision[1]),
                                                                                         np.mean(precision[2]),
                                                                                         np.mean(recall[0]),
                                                                                         np.mean(recall[1]),
                                                                                         np.mean(recall[2]),
                                                                                         time() - t2)
                print(output_str)
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (epoch_num + 1,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                print(output_str)

                
    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}

        Parameters
        ----------

        users: array of np.int64
            sequence users
        interactions: :class:`spotlight.interactions.Interactions`
            training instances, used for generate candidates
        n: int
            total number of negatives to sample for each sequence
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]

        return negative_samples

    
    def predict(self, user_id, item_ids=None):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.

        Parameters
        ----------

        user_id: int
           users id for which prediction scores needed.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        """

        sequences_np = self.test_sequence.sequences[user_id, :]
        sequences_np = np.atleast_2d(sequences_np)

        if item_ids is None:
            item_ids = np.arange(self._num_items).reshape(-1, 1)

        out = self._net.predict(self.sess,
                            sequences_np,
                            user_id,
                            item_ids)
        
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/ml1m/test/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/ml1m/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    # model arguments
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--nv', type=int, default=4)
    parser.add_argument('--nh', type=int, default=16)
    parser.add_argument('--drop', type=float, default=0.5)
    
    config = parser.parse_args()

    # set seed
    set_seed(config.seed)

    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    print(config)
    # fit model
    model = Recommender(args=config)

    model.fit(train, test, verbose=True)
