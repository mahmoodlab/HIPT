def get_knn_classification_results(dataeroot, study='tcga_lung', enc_name='vit256mean', prop=1.0):
    r"""
    Runs 10-fold CV for KNN of mean WSI embeddings
    
    Args:
        - dataroot (str): Path to mean WSI embeddings for each feature type.
        - study (str): Which TCGA study (Choices: tcga_brca, tcga_lung, tcga_kidney)
        - enc_name (str): Which encoder to use (Choices: resnet50mean, vit16mean, vit256mean)
        - prop (float): Proportion of training dataset to use
    Return:
        - aucs_knn_all (pd.DataFrame): AUCs for 10-fold CV evaluation
    """
    aucs_knn_all = {}

    for i in range(10):
        train_fname = os.path.join(dataroot, enc_name, f'{study}_{enc_name}_class_split_train_{i}.pkl')
        with open(train_fname, 'rb') as handle:
            asset_dict = pickle.load(handle)
            train_embeddings, train_labels = asset_dict['embeddings'], asset_dict['labels']
            
            if prop < 1:
                sample_inds = pd.DataFrame(range(train_embeddings.shape[0])).sample(frac=0.1, random_state=1).index
                train_embeddings = train_embeddings[sample_inds]
                train_labels = train_labels[sample_inds]

        val_fname = os.path.join(dataroot, enc_name, f'{study}_{enc_name}_class_split_test_{i}.pkl')
        with open(val_fname, 'rb') as handle:
            asset_dict = pickle.load(handle)
            val_embeddings, val_labels = asset_dict['embeddings'], asset_dict['labels']

        le = LabelEncoder().fit(train_labels)
        train_labels = le.transform(train_labels)
        val_labels = le.transform(val_labels)

        ### K-NN Evaluation
        clf = KNeighborsClassifier().fit(train_embeddings, train_labels)
        y_score = clf.predict_proba(val_embeddings)
        y_pred = clf.predict(val_embeddings)
        aucs, f1s = [], []
        if len(np.unique(val_labels)) > 2:
            for j, label in enumerate(np.unique(val_labels)):
                label_class = np.array(val_labels == label, int)
            aucs.append(sklearn.metrics.roc_auc_score(val_labels, y_score, average='macro', multi_class='ovr'))
        else:
            aucs.append(sklearn.metrics.roc_auc_score(val_labels, y_score[:,1]))
        aucs_knn_all[i] = aucs

    aucs_knn_all = pd.DataFrame(aucs_knn_all).T
    return aucs_knn_all