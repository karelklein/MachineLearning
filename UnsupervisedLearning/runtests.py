from part1 import *
from part2 import *
from silhouette import plot_silhouette
from emParamTest import gmm_selection

#================ Read Files ====================

d_xtrain, d_ytrain = readin('datasets/d_train.csv')
d_xtest, d_ytest = readin('datasets/d_test.csv')
d_x, d_y = readin('datasets/d.csv')

c_xtrain, c_ytrain = readin('datasets/c_train.csv')
c_xtest, c_ytest = readin('datasets/c_test.csv')
c_x, c_y = readin('datasets/c.csv')
#================ Part 1 =======================

# plot_silhouette(d_xtrain, d_ytrain, 2)
# plot_silhouette(c_xtrain, c_ytrain, 2)

# gmm_selection(d_xtrain)
# gmm_selection(c_xtrain)

#================ Part 2 =======================
#Do PCA
d_pca_train = pca(d_xtrain, 2)
c_pca_train = pca(c_xtrain, 2)
#np.savetxt(outputname, encoded, delimiter=',', fmt='%i')

#Do ICA
d_ica_train = ica(d_xtrain, 2)
c_ica_train = ica(c_xtrain, 2)

#Do Random
d_rp_train = randomProjection(d_xtrain, 2)
c_rp_train = randomProjection(c_xtrain, 2)

#Do SVD
d_svd_train = SVD(d_xtrain, 2)
c_svd_train = SVD(c_xtrain, 2)

# make some graphs
test_pca_components(d_xtrain)
test_pca_components(c_xtrain)

test_ica_kurtosis(d_xtrain, [3,4,5])
test_ica_kurtosis(d_xtrain, [3,4,5])

test_svd_variance(d_xtrain)
test_svd_variance(c_xtrain)

