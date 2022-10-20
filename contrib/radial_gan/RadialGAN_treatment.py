"""
Code adapted from: https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/RadialGAN
RadialGAN (Jinsung Yoon, 09/10/2018)

Inputs:
  - Train_X, Test_X: features
  - Train_M, Test_M: Mask vector (which features are selected)
  - Train_G, Test_G: Group
  - Train_Y, Test_Y: Labels
  - FSet: Which features are selected for which group

"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# %% Necessary packages
import logger as log


# %% RadialGAN
def RadialGAN_Treatment(Train_X, Train_T, Train_Y, alpha):
    """
    Train_X[0] - source
    Train_X[1] - target
    """

    # %% Preparing
    # Reset
    tf.reset_default_graph()

    # Parameters
    Dim = 64

    mb_size = 32

    Z_dim = Dim

    X1_Dim = Train_X[0].shape[1]
    X2_Dim = Train_X[1].shape[1]

    y_dim = 1
    t_dim = 1
    h_dim = Dim

    # Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

        # %% Neural Networks

    # 1. Placeholders
    # (X,y) for each group
    X1 = tf.placeholder(tf.float32, shape=[None, X1_Dim])
    t1 = tf.placeholder(tf.float32, shape=[None, t_dim])
    y1 = tf.placeholder(tf.float32, shape=[None, y_dim])

    X2 = tf.placeholder(tf.float32, shape=[None, X2_Dim])
    t2 = tf.placeholder(tf.float32, shape=[None, t_dim])
    y2 = tf.placeholder(tf.float32, shape=[None, y_dim])

    # %% Discriminator net model
    D1_W1 = tf.Variable(xavier_init([X1_Dim + t_dim + y_dim, h_dim]))
    D1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D1_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D1_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D1 = [D1_W1, D1_W2, D1_b1, D1_b2]

    D2_W1 = tf.Variable(xavier_init([X2_Dim + t_dim + y_dim, h_dim]))
    D2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D2_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D2_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D2 = [D2_W1, D2_W2, D2_b1, D2_b2]

    theta_D = theta_D1 + theta_D2

    # %% Functions
    def discriminator1(x, t, y):
        inputs = tf.concat(axis=1, values=[x, t, y])
        D1_h1 = tf.nn.tanh(tf.matmul(inputs, D1_W1) + D1_b1)
        D1_logit = tf.matmul(D1_h1, D1_W2) + D1_b2
        D1_prob = tf.nn.sigmoid(D1_logit)

        return D1_prob, D1_logit

    def discriminator2(x, t, y):
        inputs = tf.concat(axis=1, values=[x, t, y])
        D2_h1 = tf.nn.tanh(tf.matmul(inputs, D2_W1) + D2_b1)
        D2_logit = tf.matmul(D2_h1, D2_W2) + D2_b2
        D2_prob = tf.nn.sigmoid(D2_logit)

        return D2_prob, D2_logit

    # Generator Net Model (X to Z)
    G1_W1 = tf.Variable(xavier_init([X1_Dim + t_dim + y_dim, h_dim]))
    G1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G1_W2 = tf.Variable(xavier_init([h_dim, Z_dim]))
    G1_b2 = tf.Variable(tf.zeros(shape=[Z_dim]))

    theta_G1_hat = [G1_W1, G1_W2, G1_b1, G1_b2]

    G2_W1 = tf.Variable(xavier_init([X2_Dim + t_dim + y_dim, h_dim]))
    G2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G2_W2 = tf.Variable(xavier_init([h_dim, Z_dim]))
    G2_b2 = tf.Variable(tf.zeros(shape=[Z_dim]))

    theta_G2_hat = [G2_W1, G2_W2, G2_b1, G2_b2]

    # %% Function
    def generator1(x, t, y):
        inputs = tf.concat(axis=1, values=[x, t, y])
        G1_h1 = tf.nn.tanh(tf.matmul(inputs, G1_W1) + G1_b1)
        G1_log_prob = tf.matmul(G1_h1, G1_W2) + G1_b2
        G1_prob = G1_log_prob

        return G1_prob

    def generator2(x, t, y):
        inputs = tf.concat(axis=1, values=[x, t, y])
        G2_h1 = tf.nn.tanh(tf.matmul(inputs, G2_W1) + G2_b1)
        G2_log_prob = tf.matmul(G2_h1, G2_W2) + G2_b2
        G2_prob = G2_log_prob

        return G2_prob

    # %% Generator Net Model (Z to X)
    F1_W1 = tf.Variable(xavier_init([Z_dim + t_dim + y_dim, h_dim]))
    F1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    F1_W2 = tf.Variable(xavier_init([h_dim, X1_Dim]))
    F1_b2 = tf.Variable(tf.zeros(shape=[X1_Dim]))

    theta_F1 = [F1_W1, F1_W2, F1_b1, F1_b2]

    F2_W1 = tf.Variable(xavier_init([Z_dim + t_dim + y_dim, h_dim]))
    F2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    F2_W2 = tf.Variable(xavier_init([h_dim, X2_Dim]))
    F2_b2 = tf.Variable(tf.zeros(shape=[X2_Dim]))

    theta_F2 = [F2_W1, F2_W2, F2_b1, F2_b2]

    theta_G1 = theta_G1_hat + theta_F1
    theta_G2 = theta_G2_hat + theta_F2

    # %% Function
    def mapping1(z, t, y):
        inputs = tf.concat(axis=1, values=[z, t, y])
        F1_h1 = tf.nn.tanh(tf.matmul(inputs, F1_W1) + F1_b1)
        F1_log_prob = tf.matmul(F1_h1, F1_W2) + F1_b2
        F1_prob = F1_log_prob

        return F1_prob

    def mapping2(z, t, y):
        inputs = tf.concat(axis=1, values=[z, t, y])
        F2_h1 = tf.nn.tanh(tf.matmul(inputs, F2_W1) + F2_b1)
        F2_log_prob = tf.matmul(F2_h1, F2_W2) + F2_b2
        F2_prob = F2_log_prob

        return F2_prob

    # %% Structure
    # 1. Generator
    G_sample12 = mapping2(generator1(X1, t1, y1), t1, y1)
    G_sample21 = mapping1(generator2(X2, t2, y2), t2, y2)

    # 2. Discriminator
    D1_real, D1_logit_real = discriminator1(X1, t1, y1)
    D21_fake, D21_logit_fake = discriminator1(G_sample21, t2, y2)

    D2_real, D2_logit_real = discriminator2(X2, t2, y2)
    D12_fake, D12_logit_fake = discriminator2(G_sample12, t1, y1)

    # 3. Recover
    Recov_X121 = mapping1(generator2(G_sample12, t1, y1), t1, y1)
    Recov_X212 = mapping2(generator1(G_sample21, t2, y2), t2, y2)

    # Loss
    # 1. Discriminator loss
    D1_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logit_real, labels=tf.ones_like(D1_logit_real))
    )
    D21_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D21_logit_fake, labels=tf.zeros_like(D21_logit_fake))
    )
    D1_loss = D1_loss_real + D21_loss_fake

    D2_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logit_real, labels=tf.ones_like(D2_logit_real))
    )
    D12_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D12_logit_fake, labels=tf.zeros_like(D12_logit_fake))
    )
    D2_loss = D2_loss_real + D12_loss_fake

    D_loss = D1_loss + D2_loss

    # 2. Generator loss
    G21_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D21_logit_fake, labels=tf.ones_like(D21_logit_fake))
    )
    G1_loss_hat = G21_loss

    G12_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D12_logit_fake, labels=tf.ones_like(D12_logit_fake))
    )
    G2_loss_hat = G12_loss

    # Recover Loss
    G121_Recov_loss = tf.reduce_mean(tf.squared_difference(Recov_X121, X1))
    G1_Recov_loss = G121_Recov_loss

    G212_Recov_loss = tf.reduce_mean(tf.squared_difference(Recov_X212, X2))
    G2_Recov_loss = G212_Recov_loss

    G1_loss = G1_loss_hat + alpha * tf.sqrt(G1_Recov_loss)
    G2_loss = G2_loss_hat + alpha * tf.sqrt(G2_Recov_loss)

    # Solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G1_solver = tf.train.AdamOptimizer().minimize(G1_loss, var_list=theta_G1)
    G2_solver = tf.train.AdamOptimizer().minimize(G2_loss, var_list=theta_G2)

    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Iterations
    Train_X1 = Train_X[0]
    Train_T1 = Train_T[0]
    Train_Y1 = Train_Y[0]

    Train_X2 = Train_X[1]
    Train_T2 = Train_T[1]
    Train_Y2 = Train_Y[1]

    log.info(f"Training RadialGAN.")
    for it in tqdm(range(20000)):

        idx1 = np.random.permutation(len(Train_X1))
        train_idx1 = idx1[:mb_size]

        X1_mb = Train_X1[train_idx1, :]
        t1_mb = np.reshape(Train_T1[train_idx1], [mb_size, 1])
        y1_mb = np.reshape(Train_Y1[train_idx1], [mb_size, 1])

        idx2 = np.random.permutation(len(Train_X2))
        train_idx2 = idx2[:mb_size]

        X2_mb = Train_X2[train_idx2, :]
        t2_mb = np.reshape(Train_T2[train_idx2], [mb_size, 1])
        y2_mb = np.reshape(Train_Y2[train_idx2], [mb_size, 1])

        _, D_loss_curr = sess.run(
            [D_solver, D_loss], feed_dict={X1: X1_mb, t1: t1_mb, y1: y1_mb, X2: X2_mb, t2: t2_mb, y2: y2_mb}
        )
        _, G1_loss_curr = sess.run(
            [G1_solver, G1_loss], feed_dict={X1: X1_mb, t1: t1_mb, y1: y1_mb, X2: X2_mb, t2: t2_mb, y2: y2_mb}
        )
        _, G2_loss_curr = sess.run(
            [G2_solver, G2_loss], feed_dict={X1: X1_mb, t1: t1_mb, y1: y1_mb, X2: X2_mb, t2: t2_mb, y2: y2_mb}
        )

    # %%##### Data Generation
    X_sample = Train_X[0]
    T_sample = np.reshape(Train_T[0], [len(X_sample), 1])
    Y_sample = np.reshape(Train_Y[0], [len(X_sample), 1])

    new_target_samples = sess.run([G_sample12], feed_dict={X1: X_sample, t1: T_sample, y1: Y_sample})[0]

    return new_target_samples
