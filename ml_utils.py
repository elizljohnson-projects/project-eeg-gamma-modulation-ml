import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def compare_models(df, features, test_size = 0.3, seed = 325):
    """
    Compare multiple models using train/test split on trials.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: sid, group, and feature columns
    features : list
        List of feature column names
    test_size : float
        Proportion of data for test set (default 0.3)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary containing performance metrics for each model
    """    
    # prepare data
    X = df[features].values
    y = df['group'].values
    
    # split data: stratified by condition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        stratify = y,
        random_state = seed
    )
    
    print(f'Train set: {len(X_train)} trials')
    print(f'Test set: {len(X_test)} trials\n')
    
    # define pipelines for each model
    pipelines = {
        'Random Forest': Pipeline([
            ('classifier', RandomForestClassifier(
                n_estimators = 200,
                max_depth = 15,
                min_samples_split = 20,
                min_samples_leaf = 10,
                class_weight = 'balanced',
                random_state = seed,
                n_jobs = -1
            ))
        ]),
        
        'Gradient Boosting': Pipeline([
            ('classifier', GradientBoostingClassifier(
                n_estimators = 200,
                max_depth = 6,
                learning_rate = 0.1,
                random_state = seed
            ))
        ]),
        
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C = 1.0,
                class_weight = 'balanced',
                max_iter = 1000,
                random_state = seed,
                n_jobs = -1
            ))
        ]),
        
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel = 'rbf',
                C = 1.0,
                gamma = 'scale',
                class_weight = 'balanced',
                probability = True,
                random_state = seed
            ))
        ])
    }

    # get class labels for confusion matrix
    class_labels = sorted(np.unique(y))
    
    # test each model
    results = {}
    for name, pipeline in pipelines.items():
        print(f'Testing {name}...')
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average = 'weighted')
        f1_macro = f1_score(y_test, y_pred, average = 'macro')
        cm = confusion_matrix(y_test, y_pred)
        
        # store results
        results[name] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'pipeline': pipeline
        }
        
        # print results
        print(f'Accuracy:          {accuracy:.3f}')
        print(f'Balanced accuracy: {balanced_acc:.3f}')
        print(f'F1 (weighted):     {f1_weighted:.3f}')
        print(f'F1 (macro):        {f1_macro:.3f}')
        print(f'\nConfusion matrix:')
        print(f'              Predicted')
        print(f'              {class_labels[0]:<8} {class_labels[1]:<8} {class_labels[2]:<8}')
        print(f'Actual {class_labels[0]:<6} {cm[0,0]:<8} {cm[0,1]:<8} {cm[0,2]:<8}')
        print(f'       {class_labels[1]:<6} {cm[1,0]:<8} {cm[1,1]:<8} {cm[1,2]:<8}')
        print(f'       {class_labels[2]:<6} {cm[2,0]:<8} {cm[2,1]:<8} {cm[2,2]:<8}')
        print(f'\nClassification report:')
        print(classification_report(y_test, y_pred))
    
    # summary table
    print('\nSummary (ranked by balanced accuracy):')
    print(f'{"Model":<20} {"Accuracy":<9} {"Bal. acc.":<9} {"F1 (macro)":<9}')
    
    for name, res in sorted(results.items(), key = lambda x: x[1]['balanced_accuracy'], reverse = True):
        print(f'{name:<20} {res["accuracy"]:.3f}     {res["balanced_accuracy"]:.3f}     {res["f1_macro"]:.3f}')
    
    return results

def tune_gradient_boosting(df, features, test_size = 0.3, seed = 325,
                           n_iter_coarse = 30, n_iter_fine = 20):
    """
    Tune Gradient Boosting hyperparameters using 2-stage randomized search.
    Stage 1: Coarse search over wide parameter ranges
    Stage 2: Fine search narrowed around best parameters from Stage 1
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: sid, group, and feature columns
    features : list
        List of feature column names
    test_size : float
        Proportion of data for test set
    seed : int
        Random seed for reproducibility
    n_iter_coarse : int
        Number of parameter combinations for coarse search (default 30)
    n_iter_fine : int
        Number of parameter combinations for fine search (default 20)
        
    Returns:
    --------
    best_model : Pipeline
        Best performing pipeline
    search_results : dict
        Results from randomized search
    """
    # prepare data
    X = df[features].values
    y = df['group'].values
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        stratify = y,
        random_state = seed
    )
    
    print(f'Train set: {len(X_train)} trials')
    print(f'Test set: {len(X_test)} trials\n')

    # stage 1
    print(f'Stage 1: Coarse search ({n_iter_coarse} iterations, wide ranges)')
    
    # define parameter distributions
    params_coarse = {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
        'classifier__min_samples_split': randint(10, 50),
        'classifier__min_samples_leaf': randint(5, 30),
        'classifier__subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    
    # create pipeline
    pipeline = Pipeline([
        ('classifier', GradientBoostingClassifier(random_state = seed))
    ])
    
    # coarse search
    coarse_search = RandomizedSearchCV(
        pipeline,
        param_distributions = params_coarse,
        n_iter = n_iter_coarse,
        cv = 5,
        scoring = 'balanced_accuracy',
        n_jobs = -1,
        random_state = seed,
        verbose = 1
    )
    
    coarse_search.fit(X_train, y_train)

    print(f'\nBest CV score: {coarse_search.best_score_:.3f}')
    print(f'\nBest parameters from stage 1:')
    for param, value in coarse_search.best_params_.items():
        print(f'  {param}: {value}')
    
    # stage 2
    print(f'\n\nStage 2: Fine search ({n_iter_fine} iterations, narrow ranges)')

    # extract best parameters from coarse search
    best_coarse = coarse_search.best_params_
    
    # define narrow ranges around best parameters
    params_fine = {
        'classifier__n_estimators': randint(
            max(100, best_coarse['classifier__n_estimators'] - 50),
            min(500, best_coarse['classifier__n_estimators'] + 51)
        ),
        'classifier__max_depth': randint(
            max(3, best_coarse['classifier__max_depth'] - 1),
            min(10, best_coarse['classifier__max_depth'] + 2)
        ),
        'classifier__learning_rate': uniform(
            max(0.01, best_coarse['classifier__learning_rate'] - 0.05),
            0.1  # range of 0.1 around best
        ),
        'classifier__min_samples_split': randint(
            max(10, best_coarse['classifier__min_samples_split'] - 10),
            min(50, best_coarse['classifier__min_samples_split'] + 11)
        ),
        'classifier__min_samples_leaf': randint(
            max(5, best_coarse['classifier__min_samples_leaf'] - 5),
            min(30, best_coarse['classifier__min_samples_leaf'] + 6)
        ),
        'classifier__subsample': uniform(
            max(0.6, best_coarse['classifier__subsample'] - 0.1),
            min(0.2, 1.0 - (best_coarse['classifier__subsample'] - 0.1))
        ),
        'classifier__max_features': [best_coarse['classifier__max_features']]  # keep best
    }

    # fine search
    fine_search = RandomizedSearchCV(
        pipeline,
        param_distributions = params_fine,
        n_iter = n_iter_fine,
        cv = 5,
        scoring = 'balanced_accuracy',
        n_jobs = -1,
        random_state = seed + 1,  # different seed to mix things up
        verbose = 1
    )

    fine_search.fit(X_train, y_train)

    print(f'\nBest CV score: {fine_search.best_score_:.3f}')
    print(f'\nFinal best parameters:')
    for param, value in fine_search.best_params_.items():
        print(f'  {param}: {value}')

    # evaluate on test set
    best_model = fine_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average = 'macro')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f'\nTest set performance:')
    print(f'Accuracy:          {accuracy:.3f}')
    print(f'Balanced accuracy: {balanced_acc:.3f}')
    print(f'F1 (macro):        {f1_macro:.3f}')
    
    # confusion matrix
    class_labels = sorted(np.unique(y))
    print(f'\nConfusion matrix:')
    print(f'              Predicted')
    print(f'              {class_labels[0]:<8} {class_labels[1]:<8} {class_labels[2]:<8}')
    print(f'Actual {class_labels[0]:<6} {cm[0,0]:<8} {cm[0,1]:<8} {cm[0,2]:<8}')
    print(f'       {class_labels[1]:<6} {cm[1,0]:<8} {cm[1,1]:<8} {cm[1,2]:<8}')
    print(f'       {class_labels[2]:<6} {cm[2,0]:<8} {cm[2,1]:<8} {cm[2,2]:<8}')
    
    search_results = {
        'coarse_best_params': coarse_search.best_params_,
        'coarse_best_score': coarse_search.best_score_,
        'fine_best_params': fine_search.best_params_,
        'fine_best_score': fine_search.best_score_,
        'test_accuracy': accuracy,
        'test_balanced_accuracy': balanced_acc,
        'test_f1_macro': f1_macro,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'coarse_cv_results': coarse_search.cv_results_,
        'fine_cv_results': fine_search.cv_results_
    }
    
    return best_model, search_results

def rank_features(df, features, best_model):
    """
    Rank all features by importance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: sid, group, and feature columns
    features : list
        List of feature column names
        
    Returns:
    --------
    importances_df : pandas.DataFrame
        All features ranked by importance with cumulative importance
    """    
    # extract classifier from pipeline
    classifier = best_model.named_steps['classifier']
    
    # get feature importances
    importances = classifier.feature_importances_
    
    # create df with rankings
    importances_df = pd.DataFrame({
        'rank': range(1, len(features) + 1),
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending = False).reset_index(drop = True)
    
    # recalculate rank after sorting
    importances_df['rank'] = range(1, len(features) + 1)
    
    # add cumulative importance
    importances_df['cumulative_importance'] = importances_df['importance'].cumsum()
    
    # add percentage of total importance
    total_importance = importances_df['importance'].sum()
    importances_df['pct_importance'] = (importances_df['importance'] / total_importance * 100)
    importances_df['cumulative_pct'] = importances_df['pct_importance'].cumsum()
    
    print('Feature importance ranking:')
    print(f'{"Rank":<6} {"Feature":<25} {"Importance":<12} {"% Total":<10} {"Cumulative %":<15}')
    for _, row in importances_df.iterrows():
        print(f'{row["rank"]:<6} {row["feature"]:<25} {row["importance"]:<12.4f} '
              f'{row["pct_importance"]:<10.2f} {row["cumulative_pct"]:<15.2f}')
    
    # visualize   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (9, 4))
    
    # plot feature importances
    ax1.barh(importances_df['feature'], importances_df['importance'])
    ax1.set_xlabel('Importance')
    ax1.set_ylabel('Feature')
    ax1.set_title('Feature importance')
    ax1.invert_yaxis()
    
    # plot cumulative importance
    ax2.plot(importances_df['rank'], importances_df['cumulative_pct'], 
             marker = 'o', linewidth = 2)
    ax2.axhline(80, color = 'r', linestyle = '--', label = '80% threshold')
    ax2.axhline(90, color = 'orange', linestyle = '--', label = '90% threshold')
    ax2.axhline(95, color = 'g', linestyle = '--', label = '95% threshold')
    ax2.set_xlabel('Number of features')
    ax2.set_ylabel('Cumulative % of importance')
    ax2.set_title('Cumulative feature importance')
    ax2.legend()
    ax2.set_xticks(range(1, len(features) + 1))
    
    plt.tight_layout()
    plt.show()
    
    return importances_df

def test_feature_subsets(df, all_features, importances_df, best_params, 
                         test_size = 0.3, seed = 325):
    """
    Test performance with different numbers of top features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: sid, group, and feature columns
    all_features : list
        List of all feature column names
    importances_df : pandas.DataFrame
        Feature importance rankings from rank_all_features()
    best_params : dict
        Best hyperparameters from tuning (with 'classifier__' prefix)
    test_size : float
        Proportion of data for test set
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Performance metrics for different feature subset sizes
    best_n : int
        Optimal number of features (highest balanced accuracy)
    """    
    # ensure all_features is a list
    if not isinstance(all_features, list):
        all_features = list(all_features)
    
    # prepare full data
    X_full = df[all_features].values
    y = df['group'].values
    
    # split data once (same split for all feature subsets)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y,
        test_size = test_size,
        stratify = y,
        random_state = seed
    )
    
    print(f'Train set: {len(X_train_full)} trials')
    print(f'Test set: {len(X_test_full)} trials\n')
    
    # extract model params (remove 'classifier__' prefix)
    model_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
    
    # test different numbers of top features
    n_features_to_test = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    results = []
        
    for n in n_features_to_test:
        print(f'Testing top {n} features...')
        
        # select top n features
        top_features = importances_df.head(n)['feature'].tolist()
        
        # get indices of these features in all_features list
        feature_indices = [all_features.index(f) for f in top_features]
        
        # subset data to selected features
        X_train_subset = X_train_full[:, feature_indices]
        X_test_subset = X_test_full[:, feature_indices]
        
        # train model with tuned hyperparameters
        model = GradientBoostingClassifier(**model_params, random_state = seed)
        model.fit(X_train_subset, y_train)
        
        # evaluate
        y_pred = model.predict(X_test_subset)
        
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average = 'macro')
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'n_features': n,
            'features': ', '.join(top_features),
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'confusion_matrix': cm
        })
        
        print(f'  Accuracy: {accuracy:.3f}, Balanced accuracy: {balanced_acc:.3f}, F1 (macro): {f1_macro:.3f}')
        print(f'  Features: {", ".join(top_features)}')
        print()
       
    results_df = pd.DataFrame(results)
    
    # plot balanced accuracy with best highlighted    
    fig, ax = plt.subplots(1, 1, figsize = (4, 4))

    best_idx = results_df['balanced_accuracy'].idxmax()
    best_n = results_df.loc[best_idx, 'n_features']
    best_score = results_df.loc[best_idx, 'balanced_accuracy']
    
    ax.plot(results_df['n_features'], results_df['balanced_accuracy'], 
            marker = 'o', linewidth = 2, color = 'b')
    ax.scatter([best_n], [best_score], color = 'r', zorder = 5, 
               label = f'Best: {best_n} features ({best_score:.3f})')
    ax.axvline(best_n, color = 'r', linestyle = '--')
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Optimal feature subset size')
    ax.legend()
    ax.set_xticks(n_features_to_test)
    ax.set_ylim([0.55, 0.7])
    
    plt.tight_layout()
    plt.show()
    
    # print optimal configuration
    print(f'\nOptimal configuration:')
    print(f'  Number of features: {best_n}')
    print(f'  Balanced accuracy:  {best_score:.3f}')
    print(f'  Features used:')
    best_features = results_df.loc[best_idx, 'features'].split(', ')
    for i, feat in enumerate(best_features, 1):
        print(f'    {i}. {feat}')
    
    # show confusion matrix for best model
    print(f'\nConfusion matrix for best model ({best_n} features):')
    cm = results_df.loc[best_idx, 'confusion_matrix']
    class_labels = sorted(np.unique(y))
    print(f'              Predicted')
    print(f'              {class_labels[0]:<8} {class_labels[1]:<8} {class_labels[2]:<8}')
    print(f'Actual {class_labels[0]:<6} {cm[0,0]:<8} {cm[0,1]:<8} {cm[0,2]:<8}')
    print(f'       {class_labels[1]:<6} {cm[1,0]:<8} {cm[1,1]:<8} {cm[1,2]:<8}')
    print(f'       {class_labels[2]:<6} {cm[2,0]:<8} {cm[2,1]:<8} {cm[2,2]:<8}')
    
    return results_df, best_n

def permutation_test(df, selected_features, best_params, nperm = 100, 
                     test_size = 0.3, seed = 325):
    """
    Permutation test: randomly reassign subjects to conditions and retrain to verify 
    model learns condition-specific patterns rather than individual differences.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: sid, group, and feature columns
    selected_features : list
        List of selected feature names to use
    best_params : dict
        Best hyperparameters from tuning (with 'classifier__' prefix)
    nperm : int
        Number of random label shuffles to test (default 100)
    test_size : float
        Proportion of data for test set
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with real and permuted scores, p-value, and visualizations
    """    
    # prepare data with selected features
    X = df[selected_features].values
    y = df['group'].values
    sid = df['sid'].values
    
    # split data
    X_train, X_test, y_train, y_test, sid_train, sid_test = train_test_split(
        X, y, sid,
        test_size = test_size,
        stratify = y,
        random_state = seed
    )
    
    print(f'Features: {len(selected_features)}')
    print(f'Train set: {len(X_train)} trials')
    print(f'Test set: {len(X_test)} trials\n')
    
    # extract model params
    model_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}

    # create pipeline (as in compare_models)
    pipeline = Pipeline([
        ('classifier', GradientBoostingClassifier(**model_params, random_state = seed))
    ])
    
    # train model with real labels
    print('Training model with real labels...')
    pipeline.fit(X_train, y_train)
    y_pred_real = pipeline.predict(X_test)
    
    real_accuracy = accuracy_score(y_test, y_pred_real)
    real_balanced_acc = balanced_accuracy_score(y_test, y_pred_real)
    real_f1 = f1_score(y_test, y_pred_real, average = 'macro')
    
    print('\nPerformance with real labels:')
    print(f'  Accuracy:          {real_accuracy:.3f}')
    print(f'  Balanced accuracy: {real_balanced_acc:.3f}')
    print(f'  F1 (macro):        {real_f1:.3f}\n')

    # get all unique subjects in the dataset
    unique_sid = np.unique(sid)
    n_sid = len(unique_sid)
    
    # determine subjects per condition
    conditions = np.unique(y)
    n_per_cond = n_sid // 3
    
    # run permutation tests with shuffled labels
    permuted_scores = []
    
    for i in range(nperm):
        if i % round(nperm / 10) == 0:
            print(f'Processing permutation {i+1} of {nperm}...')
        
        # randomly shuffle subjects and assign to conditions
        shuffled_sid = np.random.RandomState(seed + i).permutation(unique_sid)

        # create subject-to-condition mapping
        sid_to_cond = {}
        for idx, condition in enumerate(conditions):
            start_idx = idx * n_per_cond
            end_idx = start_idx + n_per_cond if idx < 2 else n_sid
            assigned_sid = shuffled_sid[start_idx:end_idx]
            for subj in assigned_sid:
                sid_to_cond[subj] = condition

        # apply shuffled labels to training set based on subject ID
        y_train_shuffled = np.array([sid_to_cond[subj] for subj in sid_train])

        # train model with shuffled subject-to-condition mapping
        pipeline_perm = Pipeline([
            ('classifier', GradientBoostingClassifier(**model_params, random_state = seed + i))
        ])  # different seed to mix things up
        pipeline_perm.fit(X_train, y_train_shuffled)

        # evaluate on test set with real labels
        y_test_shuffled = np.array([sid_to_cond[subj] for subj in sid_test])
        y_pred_perm = pipeline_perm.predict(X_test)
        balanced_acc_perm = balanced_accuracy_score(y_test_shuffled, y_pred_perm)
        
        permuted_scores.append(balanced_acc_perm)
    
    permuted_scores = np.array(permuted_scores)
    
    # calculate p-value as proportion of permuted scores >= real score
    p_value = (permuted_scores >= real_balanced_acc).sum() / nperm
    
    # stats
    perm_mean = permuted_scores.mean()
    perm_std = permuted_scores.std()
    perm_min = permuted_scores.min()
    perm_max = permuted_scores.max()
    
    print('\nPermutation test results (balanced accuracy):')
    print(f'  Mean:   {perm_mean:.3f}')
    print(f'  SD:     {perm_std:.3f}')
    print(f'  Min:    {perm_min:.3f}')
    print(f'  Max:    {perm_max:.3f}')
    print(f'  Median: {np.median(permuted_scores):.3f}')
    print(f'\np-value: {p_value:.4f}')    
    
    # plot histogram of permuted scores with real score highlighted
    fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    
    ax.hist(permuted_scores, bins = 50, alpha = 0.7, color = 'gray',
            edgecolor = 'black', label = 'Permuted models')
    ax.axvline(real_balanced_acc, color = 'red', linewidth = 3,
               label = f'Real model: {real_balanced_acc:.3f}')
    ax.axvline(perm_mean, color = 'blue', linewidth = 2, linestyle = '--',
               label = f'Permuted mean: {perm_mean:.3f}')
    ax.axvline(1/3, color = 'orange', linewidth = 2, linestyle = ':',
               label = 'Chance: 0.333')
    ax.set_xlabel('Balanced accuracy')
    ax.set_ylabel('Density')
    ax.set_title('Real vs. permuted performance')
    ax.legend(loc = 'upper left')
    
    plt.tight_layout()
    plt.show()
    
    # store results
    results = {
        'real_accuracy': real_accuracy,
        'real_balanced_accuracy': real_balanced_acc,
        'real_f1_macro': real_f1,
        'permuted_scores': permuted_scores,
        'permuted_mean': perm_mean,
        'permuted_std': perm_std,
        'permuted_min': perm_min,
        'permuted_max': perm_max,
        'p_value': p_value,
        'nperm': nperm
    }
    
    return results
