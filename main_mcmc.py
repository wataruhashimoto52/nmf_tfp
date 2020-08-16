import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import sparse
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutineAutoBatched.Root


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


def make_recommendation_activeuser(R, prediction, user_idx, k=5):
    # get the list of actual ratings of user_idx (seen movies)
    rated_items_df_user = pd.DataFrame(R).iloc[user_idx, :]
    # get the list of predicted ratings of user_idx (unseen movies)
    user_prediction_df_user = pd.DataFrame(prediction).iloc[user_idx, :]
    # merge both lists with the movie's title
    reco_df = pd.concat(
        [rated_items_df_user, user_prediction_df_user, item_info], axis=1)
    reco_df.columns = ['rating', 'prediction', 'title']
    print("=====recommendation=====")
    print('Preferred movies for user #', user_idx)
    # returns the 5 seen movies with the best actual ratings
    print(reco_df.sort_values(by='rating', ascending=False)[:k])
    print('Recommended movies for user #', user_idx)
    reco_df = reco_df[reco_df['rating'] == 0]
    # returns the 5 unseen movies with the best predicted ratings
    print(reco_df.sort_values(by='prediction', ascending=False)[:k])
    print()
    print()

@tf.function(autograph=True)
def run_sampling(kernel, state, num_results, num_burnin_steps):
    return tfp.mcmc.sample_chain(
        kernel=kernel,
        current_state=state,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        trace_fn=lambda current_state, kernel_results: kernel_results)


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
    R_train = ConvertToDense(X_train, y_train, R_shape).astype("float64")
    R_test = ConvertToDense(X_test, y_test, R_shape).astype("float64")

    D, N = R_train.shape
    K = 20
    # model = get_nmf_model(D, N, K)
    def nmf_model_coroutine():
        qW = yield Root(tfd.Sample(tfd.Gamma(concentration=tf.cast(5., tf.float64),
        rate=tf.cast(5., tf.float64)), [D, K]))
        qH = yield Root(tfd.Sample(tfd.Gamma(concentration=tf.cast(5., tf.float64),
            rate=tf.cast(5., tf.float64)), [K, N]))
        X = yield tfd.Independent(
                tfd.Poisson(rate=tf.matmul(qW, qH)),
                reinterpreted_batch_ndims=2
            )
    model = tfd.JointDistributionCoroutineAutoBatched(nmf_model_coroutine)

    num_results = 1000
    num_burnin_steps = 100

    @tf.function(autograph=True)
    def target_log_prob(W, H):
        return model.log_prob((W, H, R_train))

    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())
    sampler = tfp.mcmc.TransformedTransitionKernel(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=tf.cast(0.1, tf.float64),
            num_leapfrog_steps=10),
        bijector=[constrain_positive, constrain_positive])

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8*num_burnin_steps),
        target_accept_prob=tf.cast(0.75, tf.float64)
    )
    nchains = 4
    W, H, _ = model.sample(nchains)
    initial_state = [W, H]

    t0 = time.time()
    samples, kernel_results = run_sampling(adaptive_sampler, initial_state,
                                           num_results, num_burnin_steps)
    t1 = time.time()

    print("Inference ran in {:.2f}s.".format(t1-t0))