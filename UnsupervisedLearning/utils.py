from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV
from scipy.stats import kurtosis
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


def load_data(data_file):
    """
    Function that reads in a csv file and returns the features and targets for
    SK-learn
    Returns a tuple, first entry is features, second is targets, third is both combined
    """
    matrix_pandas = pd.read_csv(data_file, header=None)
    matrix_np = matrix_pandas.as_matrix()
    final_col = matrix_np.shape[1] - 1
    features = matrix_np[:, :final_col]
    targets = matrix_np[:, final_col]
    return (features, targets, matrix_np)

def silhouette_analysis(feature_matrix, num_clusters, title):
    """
    Function that takes in converted dataset from numpy, and array with clusters to iterate thorugh
    """
    for i in num_clusters:
        # Create a subplot with 1 row and 1 columns
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        ax1.set_title(title)
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-0.1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(feature_matrix) + (i + 1) * 10])

        # Consider changing the seed for random state later
        # TODO: Look at the rest of the inputs for the constructor
        clusterer = KMeans(n_clusters=i, random_state=10)

        # Computer the centers of all the clusters
        cluster_centers = clusterer.fit_predict(feature_matrix)

        # Compute the average silhouette score amongst all samples
        silhouette_avg = silhouette_score(feature_matrix, cluster_centers)

        # Compute the silhouette score for all clusters in this iteration
        silhouette_values = silhouette_samples(feature_matrix, cluster_centers)

        print("For n_clusters =", i,
            "The average silhouette_score is :", silhouette_avg)

        # Sets the lowest point in 2d for each cluster in teh graph
        y_lower = 10
        for j in range(i):
            # Aggregate the silhouette scores for samples belonging to
            # cluster j, and sort them
            jth_cluster_silhouette_values = \
                silhouette_values[cluster_centers == j]

            jth_cluster_silhouette_values.sort()

            size_cluster_j = jth_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.spectral(float(j) / i)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                0, jth_cluster_silhouette_values,
                facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # plt.show()

def bic_analysis(feature_matrix, num_components, title):
    """
    Function that plots bic scores for the feature matrix iterating through the
    specified number of components
    """
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, num_components)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(feature_matrix)
            bic.append(gmm.bic(feature_matrix))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    spl = plt.subplot(1, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title(title)
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    plt.show()

def append_column(matrix, column):
    """
    Returns a copy of the matrix with a column appended to it
    """
    matrix_copy = np.array(matrix, copy=True)
    column = column.reshape((-1, 1))
    matrix_copy = np.append(matrix_copy, column, axis=1)
    return matrix_copy

def compute_optimal(loaded_data, k_array, em_range, sa_title, bic_title):
    silhouette_analysis(loaded_data[0], k_array, sa_title)
    bic_analysis(loaded_data[0], em_range, bic_title)

def fit_cluster_data(train_data, test_data, num_kcluster, num_em_components, cov_type,
                    kmeans_center_name, kmeans_train_name, kmeans_test_name,
                    em_train_name, em_test_name, int_bool=False):
    k_clusterer = KMeans(n_clusters=num_kcluster, random_state=0)
    k_clusterer.fit(train_data[0])
    k_labels_train = k_clusterer.labels_
    k_labels_test = k_clusterer.predict(test_data[0])

    em_clusterer = GaussianMixture(n_components=num_em_components,
        init_params='kmeans', covariance_type=cov_type)
    em_clusterer.fit(train_data[0])
    em_labels_train = em_clusterer.predict(train_data[0])
    em_labels_test = em_clusterer.predict(test_data[0])

    k_train_feature = append_column(train_data[0], k_labels_train)
    k_train_labels = append_column(k_train_feature, train_data[1])

    k_test_feature = append_column(test_data[0], k_labels_test)
    k_test_labels = append_column(k_test_feature, test_data[1])

    em_train_feature = append_column(train_data[0], em_labels_train)
    em_train_labels = append_column(em_train_feature, train_data[1])

    em_test_feature = append_column(test_data[0], em_labels_test)
    em_test_labels = append_column(em_test_feature, test_data[1])

    np.savetxt(kmeans_center_name, k_clusterer.cluster_centers_, delimiter=',')

    if int_bool:
        np.savetxt(kmeans_train_name, k_train_labels, fmt='%i', delimiter=',')
        np.savetxt(kmeans_test_name, k_test_labels, fmt='%i', delimiter=',')
        np.savetxt(em_train_name, em_train_labels, fmt='%i', delimiter=',')
        np.savetxt(em_test_name, em_test_labels, fmt='%i', delimiter=',')
    # else:
    #     np.savetxt(kmeans_name, k_results_labels, delimiter=',')
    #     np.savetxt(em_name, k_results_labels, delimiter=',')

def fit_cluster_data22(loaded_data, num_kcluster, num_em_components,
                    kmeans_name, em_name, kmeans_center_name, kmeans_train_name=None, kmeans_test_name=None,
                    em_train_name=None, em_test_name=None, int_bool=False,
                    use_train_test=False):
    k_clusterer = KMeans(n_clusters=num_kcluster, random_state=0)
    k_clusterer.fit(loaded_data[0])
    k_labels = k_clusterer.labels_

    em_clusterer = GaussianMixture(n_components=num_em_components, init_params='kmeans')
    em_clusterer.fit(loaded_data[0])
    em_labels = em_clusterer.predict(loaded_data[0])

    k_results_feature = append_column(loaded_data[0], k_labels)
    k_results_labels = append_column(k_results_feature, loaded_data[1])
    em_results_feature = append_column(loaded_data[0], em_labels)
    em_results_labels = append_column(em_results_feature, loaded_data[1])

    np.savetxt(kmeans_center_name, k_clusterer.cluster_centers_, delimiter=',')

    if int_bool:
        np.savetxt(kmeans_name, k_results_labels, fmt='%i', delimiter=',')
        np.savetxt(em_name, em_results_labels, fmt='%i', delimiter=',')
    else:
        np.savetxt(kmeans_name, k_results_labels, delimiter=',')
        np.savetxt(em_name, k_results_labels, delimiter=',')

    if use_train_test:
        kmeans_train, kmeans_test = split_set(k_results_feature, loaded_data[1], 0.8)
        em_train, em_test = split_set(em_results_feature, loaded_data[1], 0.8)
        if int_bool:
            np.savetxt(kmeans_name, kmeans_train, fmt='%i', delimiter=',')
            np.savetxt(kmeans_name, kmeans_test, fmt='%i', delimiter=',')
            np.savetxt(em_name, em_train, fmt='%i', delimiter=',')
            np.savetxt(em_name, em_test, fmt='%i', delimiter=',')
        else:
            np.savetxt(kmeans_name, kmeans_train, delimiter=',')
            np.savetxt(kmeans_name, kmeans_test, delimiter=',')
            np.savetxt(em_name, em_train, delimiter=',')
            np.savetxt(em_name, exm_test, delimiter=',')

def fit_transform_data(loaded_data, num_pca_components, num_ica_components,
                        num_rp_components, num_svd_components, pca_name, pca_train_name, pca_test_name,
                        ica_name, ica_train_name, ica_test_name,
                        rp_name, rp_train_name, rp_test_name,
                        svd_name, svd_train_name, svd_test_name):
    # Train and fit PCA
    pca_transformer = PCA(n_components=num_pca_components)
    pca_fitted = pca_transformer.fit_transform(loaded_data[0])

    # Train and fit ICA
    ica_transformer = FastICA(n_components=num_ica_components)
    ica_fitted = ica_transformer.fit_transform(loaded_data[0])

    # Train and fit RP
    # TODO: Iterate through more
    rp_transformer = GaussianRandomProjection(n_components=2)
    rp_fitted = rp_transformer.fit_transform(loaded_data[0])

    svd_transformer = TruncatedSVD(n_components=num_svd_components)
    svd_fitted = svd_transformer.fit_transform(loaded_data[0])

    # Train and fit other

    # Split the fitted sets to be saved as training/test sets
    pca_fitted_train, pca_fitted_test = split_set(pca_fitted, loaded_data[1], 0.8)
    ica_fitted_train, ica_fitted_test = split_set(ica_fitted, loaded_data[1], 0.8)
    rp_fitted_train, rp_fitted_test = split_set(rp_fitted, loaded_data[1], 0.8)
    svd_fitted_train, svd_fitted_test = split_set(svd_fitted, loaded_data[1], 0.8)
    pca_fitted = append_column(pca_fitted, loaded_data[1])
    ica_fitted = append_column(ica_fitted, loaded_data[1])
    rp_fitted = append_column(rp_fitted, loaded_data[1])
    svd_fitted = append_column(svd_fitted, loaded_data[1])

    # Save the datasets
    np.savetxt(pca_name, pca_fitted, delimiter=',')
    np.savetxt(pca_train_name, pca_fitted_train, delimiter=',')
    np.savetxt(pca_test_name, pca_fitted_test, delimiter=',')

    np.savetxt(ica_name, ica_fitted, delimiter=',')
    np.savetxt(ica_train_name, ica_fitted_train, delimiter=',')
    np.savetxt(ica_test_name, ica_fitted_test, delimiter=',')

    np.savetxt(rp_name, rp_fitted, delimiter=',')
    np.savetxt(rp_train_name, rp_fitted_train, delimiter=',')
    np.savetxt(rp_test_name, rp_fitted_test, delimiter=',')

    np.savetxt(svd_name, svd_fitted, delimiter=',')
    np.savetxt(svd_train_name, svd_fitted_train, delimiter=',')
    np.savetxt(svd_test_name, svd_fitted_test, delimiter=',')

def split_set(X, y, train_split):
    num_instances = len(X)
    cut_off = int(train_split * num_instances)
    X_train = X[:cut_off]
    X_test = X[cut_off:]
    y_train = y[:cut_off]
    y_test = y[cut_off:]
    train_set = append_column(X_train, y_train)
    test_set = append_column(X_test, y_test)
    return train_set, test_set

def fit_nn_data(train_data, test_data, hidden_layer_tuple, batch=100,
                learn_rate=0.3, iterations=100, momentum_rate=0.2):
    nn = MLPClassifier(hidden_layer_sizes=hidden_layer_tuple, solver='sgd',
        batch_size=batch, learning_rate_init=learn_rate, max_iter=iterations,
        momentum=momentum_rate)
    start = time.clock()
    nn.fit(train_data[0], train_data[1])
    elapsed = (time.clock() - start)
    error = nn.score(test_data[0], test_data[1])
    print("Accuracy is " + str(error))
    print("Training time is " + str(elapsed))
    return error, elapsed

def optimize_nn_data(train_data, lc_title, vc_title, param_range, hidden_layer_tuple,
                    batch=100, learn_rate=0.3, iterations=1000, momentum_rate=0.2):
    # NOTE: Look for accuracy, learning curve, error per iteration
    nn = MLPClassifier(solver='sgd', batch_size=batch,
        learning_rate_init=learn_rate, max_iter=iterations,
        momentum=momentum_rate)
    plot_learning_curve(nn, lc_title, train_data[0], train_data[1])
    plot_validation_curve(nn, vc_title, train_data[0], train_data[1],
        "max_iter", param_range)

def grid_nn_data(train_data, grid_name, param_grid, batch=100, learn_rate=0.3,
                iterations=1000, momentum_rate=0.2):
    nn = MLPClassifier(solver='sgd', batch_size=batch,
        learning_rate_init=learn_rate, max_iter=iterations,
        momentum=momentum_rate)
    grid_search(nn, param_grid, train_data[0], train_data[1], grid_name)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=10,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    percentage_data = train_sizes
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("% of data")
    plt.ylabel("% Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.xticks(train_sizes, percentage_data)
    plt.show()

def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=10):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("% Error")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def grid_search(estimator, param_grid, X, y, name):
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=10)
    grid.fit(X, y)
    results = pd.DataFrame(grid.cv_results_)
    results.to_csv(name)

def create_hidden_layer_array(loaded_data, num_layers):
    num_features = loaded_data[0].shape[1]
    # TODO: Let this func. accept multivariate data later
    num_targets = 1
    sum_total = num_features + num_targets
    mean_total = int((num_features + num_targets) / 2)
    hidden_layers = []

    nf_tuple = ()
    nt_tuple = ()
    st_tuple = ()
    mt_tuple = ()
    for i in range(0, num_layers):
        nf_tuple = nf_tuple + (num_features,)
        nt_tuple = nt_tuple + (num_targets,)
        st_tuple = st_tuple + (sum_total,)
        mt_tuple = mt_tuple + (mean_total,)
        hidden_layers.append(nf_tuple)
        hidden_layers.append(nt_tuple)
        hidden_layers.append(st_tuple)
        hidden_layers.append(mt_tuple)

    hidden_layers = list(set(hidden_layers))
    return hidden_layers

def test_pca_components(x_train):
    pca2 = PCA()
    pca2.fit_transform(x_train)
    eigenvalues = pca2.explained_variance_
    
    n_features = np.linspace(1,len(x_train[0]), len(x_train[0]))
    plt.plot(n_features, eigenvalues, linewidth=3)
    plt.title('Scree Test for PCA')
    plt.xlabel('Number of Eigenvalues')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def test_ica_kurtosis(x_train, n_components_arr):
    fig, axes = plt.subplots(1, len(n_components_arr))
    c = 0
    for n in n_components_arr:
        ica1 = FastICA(n_components=n)
        matrix = ica1.fit_transform(x_train)
        n_features = len(matrix[0])
        cols = []
        for i in range(n_features):
            cols.append([row[i] for row in matrix])
        kurtoses = [kurtosis(feature_col) for feature_col in cols]
        numfeatures = range(len(kurtoses))
        
        axes[c].bar(numfeatures, kurtoses, alpha=0.4)
        axes[c].set_title('%i components' %n)
        axes[c].set_xlabel('Feature')
        axes[0].set_ylabel('Kurtosis')
        axes[c].set_xticks(numfeatures)
        axes[c].grid(True)
        c += 1
    plt.grid(True)
    plt.show()

def test_svd_variance(x_train):
    n_components = len(x_train[0])
    total_ev = []
    for n in range(1, n_components):
        svd = TruncatedSVD(n_components=n)
        svd.fit_transform(x_train)
        evr = svd.explained_variance_ratio_
        total_ev.append(evr.sum())

    x = np.linspace(1,len(total_ev), len(total_ev))
    plt.plot(x, total_ev, linewidth=3, color='red')
    plt.ylabel('Total Variance Explained')
    plt.xlabel('Number of Components Used')
    plt.xticks(x)
    plt.title('Explained Variance using SVD')
    plt.grid(True)
    plt.show()