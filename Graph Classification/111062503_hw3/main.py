from argparse import ArgumentParser
from data_loader import load_data
import torch
import torch.nn as nn

from model import Grace
from aug import aug
# from model import YourGNNModel # Build your model in model.py

import numpy as np
import functools

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from tqdm import tqdm
    
import os
import warnings
warnings.filterwarnings("ignore")
def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret
def label_classification(embeddings, y_train, train_mask, test_mask):
    X = embeddings.detach().cpu().numpy()
    # for Y in [y_train,y_test]:
    #     Y = Y.detach().cpu().numpy()
    #     Y = Y.reshape(-1, 1)
        # onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
        # Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
    y_train=y_train.detach().cpu().numpy()
    y_train=y_train.reshape(-1,1)
    X = normalize(X, norm='l2')
    X_train = X[train_mask]
    X_test = X[test_mask]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)
    return y_pred


if __name__ == '__main__':
    # Step 1: Load hyperparameters =================================================================== #

    drop_edge_rate_1 = 0.3 # Drop edge ratio of the 1st augmentation.
    drop_edge_rate_2 = 0.3 # Drop edge ratio of the 2nd augmentation.
    drop_feature_rate_1 = 0.3 # Drop feature ratio of the 1st augmentation.
    drop_feature_rate_2 = 0.3 # Drop feature ratio of the 2nd augmentation.
    temp = 0.9 # Temperature.
    num_layers=2 # number of GNN layers
    act_fns='relu'
    act_fn = ({'relu': nn.ReLU(), 'prelu': nn.PReLU()})[act_fns]
    epochs=250


    # Step 2: Prepare data =================================================================== #
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()

    in_size = features.shape[1]

    # out_size = num_classes
    out_size=256
    print("in_size ",in_size)
    print("\nout_size ",out_size)
    hid_size=256
    print('train labels\n')
    print(train_labels)
    # Step 3: Create model =================================================================== #
    model = Grace(in_size, hid_size, out_size, num_layers, act_fn, temp)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print("Training...")
    print("test_mask\n",test_mask)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, features, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, features, drop_feature_rate_2, drop_edge_rate_2)

        loss = model(graph1, graph2, feat1, feat2)
        loss.backward()
        optimizer.step()

        print(f'Epoch={epoch:03d}, loss={loss.item():.4f}') 
    print("Testing...")
    
    graph = graph.add_self_loop()
    embeds = model.get_embedding(graph, features)
    #indices=label_classification(embeds, train_labels,test_labels, train_mask, test_mask)
    indices=label_classification(embeds, val_labels, val_mask, test_mask)
    
    
    print("Export predictions as csv file.")
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(indices):
            for i in range(len(pred)):
                if pred[i]==True:
                    f.write(f'{idx},{int(i)}\n')

