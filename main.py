import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import sparse
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

tfd = tfp.distributions
tfb = tfp.bijectors
optimizer = tf.keras.optimizers.Adam(lr=0.05)


def GetShape(filename):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(filename, sep='\t', names=names)
    n_users = len(df['user_id'].unique())
    n_items = len(df['item_id'].unique())
    return (n_users, n_items)


def LoadData(filename, R_shape):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(filename, sep='\t', names=names)
    X = df[['user_id', 'item_id']].values
    y = df['rating'].values
    return X, y, ConvertToDense(X, y, R_shape)


def ConvertToDense(X, y, shape):
    row = X[:, 0]
    col = X[:, 1]
    data = y
    matrix_sparse = sparse.csr_matrix(
        (data, (row, col)), shape=(shape[0]+1, shape[1]+1))
    R = matrix_sparse.todense()
    R = R[1:, 1:]
    R = np.asarray(R)
    return R

def get_rmse(pred, actual):
        pred = pred[actual.nonzero()].flatten()     # Ignore nonzero terms
        actual = actual[actual.nonzero()].flatten()  # Ignore nonzero terms
        return np.sqrt(mean_squared_error(pred, actual))


@tf.function(autograph=True)
def train_step(model, data, D):
    with tf.GradientTape() as tape:
        log_likelihoods, kl_sum = model(data)
        elbo_loss = kl_sum/D - tf.reduce_mean(log_likelihoods)

    gradients = tape.gradient(elbo_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return elbo_loss


def make_recommendation_activeuser(R, prediction, user_idx, k=5):
    rated_items_df_user = pd.DataFrame(R).iloc[user_idx, :]                 # get the list of actual ratings of user_idx (seen movies)
    user_prediction_df_user = pd.DataFrame(prediction).iloc[user_idx,:]     # get the list of predicted ratings of user_idx (unseen movies)
    reco_df = pd.concat([rated_items_df_user, user_prediction_df_user, item_info], axis=1)   # merge both lists with the movie's title
    reco_df.columns = ['rating','prediction','title']
    print("=====recommendation=====")
    print('Preferred movies for user #', user_idx)
    print(reco_df.sort_values(by='rating', ascending=False)[:k] )           # returns the 5 seen movies with the best actual ratings
    print('Recommended movies for user #', user_idx)
    reco_df = reco_df[ reco_df['rating'] == 0 ]
    print(reco_df.sort_values(by='prediction', ascending=False)[:k] )        # returns the 5 unseen movies with the best predicted ratings
    print()
    print()


class NMFModel(tf.keras.Model):
    """
    A Bayesian Non-negative Matrix Factorization Model using Variational Inference.

    """

    def __init__(self, D, N, K):
        super(NMFModel, self).__init__()
        self.D = D
        self.N = N
        self.K = K
        self.a_W = tf.Variable(tf.random.gamma((D, K), 5., 5.), constraint=lambda t: tf.clip_by_value(
            t, 0.01 * tf.ones((D, K)), 100. * tf.ones((D, K))))
        self.b_W = tf.Variable(tf.random.gamma((D, K), 5., 5.), constraint=lambda t: tf.clip_by_value(
            t, 0.01 * tf.ones((D, K)), 100. * tf.ones((D, K))))
        self.a_H = tf.Variable(tf.random.gamma((K, N), 5., 5.), constraint=lambda t: tf.clip_by_value(
            t, 0.01 * tf.ones((K, N)), 100. * tf.ones((K, N))))
        self.b_H = tf.Variable(tf.random.gamma((K, N), 5., 5.), constraint=lambda t: tf.clip_by_value(
            t, 0.01 * tf.ones((K, N)), 100. * tf.ones((K, N))))
        self.W_prior = tfd.Gamma(concentration=1./self.K, rate=1./self.K)
        self.H_prior = tfd.Gamma(concentration=1./self.K, rate=1./self.K)

    def call(self, x, sampling=False):
        W = tfd.Gamma(concentration=self.a_W, rate=self.b_W)
        H = tfd.Gamma(concentration=self.a_H, rate=self.b_H)

        if sampling:
            raise NotImplementedError
        else:
            W_sample = W.mean()
            H_sample = H.mean()

        density = tfd.Poisson(rate=tf.matmul(W_sample, H_sample))

        log_likelihoods = density.log_prob(x)

        W_div = tf.reduce_sum(tfd.kl_divergence(W, self.W_prior))
        H_div = tf.reduce_sum(tfd.kl_divergence(H, self.H_prior))
        kl_sum = W_div + H_div

        return log_likelihoods, kl_sum


if __name__ == "__main__":
    # Loading ratings
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv('data/ml-100k/u.data',
                             sep='\t', names=names, encoding="ISO-8859-1",)

    # Loading movies info
    # Information about the items (keeps only movie's name)
    item_info = pd.read_csv('data/ml-100k/u.item', sep='|',
                            header=None, usecols=[1], encoding="ISO-8859-1",)
    item_info.columns = ['title']
    n_users = len(ratings_df['user_id'].unique())
    n_items = len(ratings_df['item_id'].unique())
    R_shape = (n_users, n_items)

    R_shape = GetShape('data/ml-100k/u.data')
    X, y, R = LoadData('data/ml-100k/u.data', R_shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    R_train = ConvertToDense(X_train, y_train, R_shape).astype("float32")
    R_test = ConvertToDense(X_test, y_test, R_shape).astype("float32")

    D, N = R_train.shape
    K = 20
    model = NMFModel(D, N, K)

    # train
    epochs = 5000
    eps = 0.00001
    elbos = []
    R_train = R_train.astype("float32")
    R_test = R_test.astype("float32")
    for epoch in range(epochs):
        elbo = train_step(model, R_train, D)
        elbos.append(elbo.numpy())

        if epoch % 20 == 0:
            print(elbo.numpy())

        if epoch > 10 and abs(elbo - elbos[epoch-1]) < eps:
            break


    # get predictive distribution
    qW = tfd.Gamma(model.a_W, model.b_W)
    qH = tfd.Gamma(model.a_H, model.b_H)

    Ws = [qW.sample() for _ in range(100)]
    Hs = [qH.sample() for _ in range(100)]

    Rs = []
    for i in range(100):
        d = tfd.Poisson(rate=tf.matmul(Ws[i], Hs[i]))
        Rs.append(d.sample())

    R_pred = tf.reduce_mean(tf.stack(Rs), axis=0).numpy()
    R_pred[R_pred > 5] = 5.
    R_pred[R_pred < 1] = 1.

    print("RMSE (test)", get_rmse(R_pred, R_test))

    make_recommendation_activeuser(R, R_pred, user_idx=50, k=5)

    tf.keras.models.save_model(model=model, filepath="models/nmf.model")
