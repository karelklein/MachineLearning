"""
module to hold constants for all parts of the project
"""

# Original datasets
D_TRAIN_CSV = "datasets/d_train.csv"
D_TEST_CSV = "datasets/d_test.csv"
D_CSV = "datasets/d.csv"

C_TRAIN_CSV = "datasets/c_train.csv"
C_TEST_CSV = "datasets/c_test.csv"
C_CSV = "datasets/c.csv"

# Clustered original datasets (part one results)
D_KMEANS_CSV = "p1_results/d_kmeans.csv"
D_EM_CSV = "p1_results/d_em.csv"
C_KMEANS_CSV = "p1_results/c_kmeans.csv"
C_EM_CSV = "p1_results/c_em.csv"

D_KMEANS_CENTER_CSV = "p1_results/d_kmeans_center.csv"
C_KMEANS_CENTER_CSV = "p1_results/c_kmeans_center.csv"

# Transformed datasets (part two results)
D_PCA_CSV = "transformed_datasets/d_pca.csv"
D_PCA_TRAIN_CSV = "transformed_datasets/d_pca_train.csv"
D_PCA_TEST_CSV = "transformed_datasets/d_pca_test.csv"

D_ICA_CSV = "transformed_datasets/d_ica.csv"
D_ICA_TRAIN_CSV = "transformed_datasets/d_ica_train.csv"
D_ICA_TEST_CSV = "transformed_datasets/d_ica_test.csv"

D_RP_CSV = "transformed_datasets/d_rp.csv"
D_RP_TRAIN_CSV = "transformed_datasets/d_rp_train.csv"
D_RP_TEST_CSV = "transformed_datasets/d_rp_test.csv"

D_SVD_CSV = "transformed_datasets/d_svd.csv"
D_SVD_TRAIN_CSV = "transformed_datasets/d_svd_train.csv"
D_SVD_TEST_CSV = "transformed_datasets/d_svd_test.csv"

C_PCA_CSV = "transformed_datasets/c_pca.csv"
C_PCA_TRAIN_CSV = "transformed_datasets/c_pca_train.csv"
C_PCA_TEST_CSV = "transformed_datasets/c_pca_test.csv"

C_ICA_CSV = "transformed_datasets/c_ica.csv"
C_ICA_TRAIN_CSV = "transformed_datasets/c_ica_train.csv"
C_ICA_TEST_CSV = "transformed_datasets/c_ica_test.csv"

C_RP_CSV = "transformed_datasets/c_rp.csv"
C_RP_TRAIN_CSV = "transformed_datasets/c_train_rp.csv"
C_RP_TEST_CSV = "transformed_datasets/c_test_rp.csv"

C_SVD_CSV = "transformed_datasets/c_svd.csv"
C_SVD_TRAIN_CSV = "transformed_datasets/c_svd_train.csv"
C_SVD_TEST_CSV = "transformed_datasets/c_svd_test.csv"

# Clustered transformed datasets (part three results)
D_KMEANS_PCA_CSV = "p3_results/d_kmeans_pca.csv"
D_KMEANS_PCA_TRAIN_CSV = "p3_results/d_kmeans_pca_train.csv"
D_KMEANS_PCA_TEST_CSV = "p3_results/d_kmeans_pca_test.csv"
D_KMEANS_PCA_CENTER_CSV = "p4_results/d_kmeans_pca_center.csv"

D_EM_PCA_CSV = "p3_results/d_em_pca.csv"
D_EM_PCA_TRAIN_CSV = "p3_results/d_em_pca_train.csv"
D_EM_PCA_TEST_CSV = "p3_results/d_em_pca_test.csv"

D_KMEANS_ICA_CSV = "p3_results/d_kmeans_ica.csv"
D_KMEANS_ICA_TRAIN_CSV = "p3_results/d_kmeans_ica_train.csv"
D_KMEANS_ICA_TEST_CSV = "p3_results/d_kmeans_ica_test.csv"
D_KMEANS_ICA_CENTER_CSV = "p3_results/d_kmeans_ica_center.csv"

D_EM_ICA_CSV = "p3_results/d_em_ica.csv"
D_EM_ICA_TRAIN_CSV = "p3_results/d_em_ica_train.csv"
D_EM_ICA_TEST_CSV = "p3_results/d_em_ica_test.csv"

D_KMEANS_RP_CSV = "p3_results/d_kmeans_rp.csv"
D_KMEANS_RP_TRAIN_CSV = "p3_results/d_kmeans_rp_train.csv"
D_KMEANS_RP_TEST_CSV = "p3_results/d_kmeans_rp_test.csv"
D_KMEANS_RP_CENTER_CSV = "p3_results/d_kmeans_rp_center.csv"

D_EM_RP_CSV = "p3_results/d_em_ica.csv"
D_EM_RP_TRAIN_CSV = "p3_results/d_em_rp_train.csv"
D_EM_RP_TEST_CSV = "p3_results/d_em_rp_test.csv"

D_KMEANS_SVD_CSV = 'p3_results/d_kmeans_svd.csv'
D_KMEANS_SVD_TRAIN_CSV = 'p3_results/d_kmeans_svd_train.csv'
D_KMEANS_SVD_TEST_CSV = 'p3_results/d_kmeans_svd_test.csv'
D_KMEANS_SVD_CENTER_CSV = "p3_results/d_kmeans_svd_center.csv"

D_EM_SVD_CSV = 'p3_results/d_em_svd.csv'
D_EM_SVD_TRAIN_CSV = 'p3_results/d_em_svd_train.csv'
D_EM_SVD_TEST_CSV = 'p3_results/d_em_svd_test.csv'

C_KMEANS_PCA_CSV = "p3_results/c_kmeans_pca.csv"
C_KMEANS_PCA_TRAIN_CSV = "p3_results/c_kmeans_pca_train.csv"
C_KMEANS_PCA_TEST_CSV = "p3_results/c_kmeans_pca_test.csv"
C_KMEANS_PCA_CENTER_CSV = "p3_results/c_kmeans_pca_center.csv"

C_EM_PCA_CSV = "p3_results/c_em_pca.csv"
C_EM_PCA_TRAIN_CSV = "p3_results/c_em_pca_train.csv"
C_EM_PCA_TEST_CSV = "p3_results/c_em_pca_test.csv"

C_KMEANS_ICA_CSV = "p3_results/c_kmeans_ica.csv"
C_KMEANS_ICA_TRAIN_CSV = "p3_results/c_kmeans_ica_train.csv"
C_KMEANS_ICA_TEST_CSV = "p3_results/c_kmeans_ica_test.csv"
C_KMEANS_ICA_CENTER_CSV = "p3_results/c_kmeans_ica_center.csv"

C_EM_ICA_CSV = "p3_results/c_em_ica.csv"
C_EM_ICA_TRAIN_CSV = "p3_results/c_em_ica_train.csv"
C_EM_ICA_TEST_CSV = "p3_results/c_em_ica_test.csv"

C_KMEANS_RP_CSV = "p3_results/c_kmeans_rp.csv"
C_KMEANS_RP_TRAIN_CSV = "p3_results/c_kmeans_rp_train.csv"
C_KMEANS_RP_TEST_CSV = "p3_results/c_kmeans_rp_test.csv"
C_KMEANS_RP_CENTER_CSV = "p3_results/c_kmeans_rp_center.csv"

C_EM_RP_CSV = "p3_results/c_em_ica.csv"
C_EM_RP_TRAIN_CSV = "p3_results/c_em_rp_train.csv"
C_EM_RP_TEST_CSV = "p3_results/c_em_rp_test.csv"


C_KMEANS_SVD_CSV = 'p3_results/d_kmeans_svd.csv'
C_KMEANS_SVD_TRAIN_CSV = 'p3_results/d_kmeans_svd_train.csv'
C_KMEANS_SVD_TEST_CSV = 'p3_results/d_kmeans_svd_test.csv'
C_KMEANS_SVD_CENTER_CSV = "p3_results/c_kmeans_svd_center.csv"

C_EM_SVD_CSV = 'p3_results/c_em_svd.csv'
C_EM_SVD_TRAIN_CSV = 'p3_results/c_em_svd_train.csv'
C_EM_SVD_TEST_CSV = 'p3_results/c_em_svd_test.csv'

# Grid results (part four results)
D_PCA_GRID_CSV = "p4_results/d_pca_grid.csv"
D_ICA_GRID_CSV = "p4_results/d_ica_grid.csv"
D_RP_GRID_CSV = "p4_results/d_rp_grid.csv"
D_SVD_GRID_CSV = "p4_results/d_svd_grid.csv"

#Grid results (part five results)
D_KMEANS_PCA_GRID_CSV = "p4_results/d_kmeans_pca_grid.csv"
D_EM_PCA_GRID_CSV = "p4_results/d_em_pca_grid.csv"

D_KMEANS_ICA_GRID_CSV = "p4_results/d_kmeans_ica_grid.csv"
D_EM_ICA_GRID_CSV = "p4_results/d_em_ica_grid.csv"

D_KMEANS_RP_GRID_CSV = "p4_results/d_kmeans_rp_grid.csv"
D_EM_RP_GRID_CSV = "p4_results/d_em_rp_grid.csv"

D_KMEANS_SVD_GRID_CSV = "p4_results/d_kmeans_svd_grid.csv"
D_EM_SVD_GRID_CSV = "p4_results/d_em_svd_grid.csv"

