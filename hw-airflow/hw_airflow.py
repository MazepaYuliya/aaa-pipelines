import os.path
import pandas as pd


import mlflow
import json
import numpy as np
import os
import pandas as pd
import random 

import implicit
# from lightfm import LightFM
# from lightfm.data import Dataset as LFM_Dataset
# from lightfm.evaluation import precision_at_k, recall_at_k
import mlflow
import torch
import torch.optim as optim
from scipy.sparse import coo_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import argparse


SEED = 42

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(SEED)


class RecDataset(Dataset):
    def __init__(self, users, items, item_per_users):
        self.users = users
        self.items = items
        self.item_per_users=item_per_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        user = self.users[i]
        return torch.tensor(user), torch.tensor(self.items[i]), self.item_per_users[user]


class LatentFactorModel(nn.Module):
    def __init__(self, edim, user_indexes, node_indexes):
        super(LatentFactorModel, self).__init__()
        self.edim = edim
        self.users = nn.Embedding(max(user_indexes) + 1, edim)
        self.items = nn.Embedding(max(node_indexes) + 1, edim)

    def forward(self, users, items):
        user_embedings = self.users(users).reshape(-1, self.edim )
        item_embedings = self.items(items)
        res = torch.einsum('be,bne->bn', user_embedings, item_embedings)
        return res 

    def pred_top_k(self, users, K=10):
        user_embedings = self.users(users).reshape(-1, self.edim )
        item_embedings = self.items.weight
        res = torch.einsum('ue,ie->ui', user_embedings, item_embedings)
        return torch.topk(res, K, dim=1)


def collate_fn(batch, num_negatives, num_items):
    users, target_items, users_negatives = [],[], []
    for triplets in batch:
        user, target_item, seen_item = triplets
        
        users.append(user)
        target_items.append(target_item)
        user_negatives = []
        
        while len(user_negatives)< num_negatives:
            candidate = random.randint(0, num_items)
            if candidate not in seen_item:
                user_negatives.append(candidate)
                
        users_negatives.append(user_negatives)

    positive = torch.ones(len(batch), 1)       
    negatives = torch.zeros(len(batch), num_negatives)
    labels = torch.hstack([positive, negatives])
    items = torch.hstack([torch.tensor(target_items).reshape(-1, 1), torch.tensor(users_negatives)])
    return torch.hstack(users), items, labels


def calc_hitrate(df_preds, K):
    return  df_preds[df_preds['rank']<K].groupby('user_index')['relevant'].max().mean()


def calc_prec(df_preds, K):
    return  (df_preds[df_preds['rank']<K].groupby('user_index')['relevant'].mean()).mean()


def fit_predict_latent_factory_model(df, df_train, df_test, **params):
    K = params['k']

    user2seen = df_train.groupby('user_index')['node_index'].agg(lambda x: list(set(x)))
    test_users = df_test['user_index'].unique()

    train_dataset = RecDataset(df_train['user_index'].values, df_train['node_index'], user2seen)
    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=0,
        batch_size=params['batch_size'],
        collate_fn=lambda x: collate_fn(
            x, params['num_negatives'], max(df['node_index'].values)
        )
    )

    model = LatentFactorModel(params['edim'], df['user_index'], df['node_index'])
    
    optimizer_constructor = getattr(optim, params['optimizer'])
    optimizer = optimizer_constructor(model.parameters(), params['lr'])

    for _ in range(params['epoch']):
        losses = []
        for i in dataloader:
            users, items, labels = i
            optimizer.zero_grad()
            logits = model(users, items)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    preds = model.pred_top_k(torch.tensor(test_users), K)[1].numpy()
    df_preds = pd.DataFrame(
        {
            'node_index': list(preds),
            'user_index': test_users,
            'rank': [[j for j in range(0, K)]for i in range(len(preds))]
        }
    )

    df_preds = df_preds.explode(['node_index', 'rank']).merge(
        df_test[['user_index', 'node_index']].assign(relevant=1).drop_duplicates(),
        on = ['user_index', 'node_index'],
        how='left' ,
    )
    df_preds['relevant'] = df_preds['relevant'].fillna(0)
    
    prec_30 = calc_prec(df_preds, K)
    hitrate = calc_hitrate(df_preds, K)

    model_keys = [
        'batch_size', 'num_negatives', 'edim', 'epoch', 'lr', 'optimizer'
    ]
    model_params = {key: params[key] for key in params if key in model_keys}
    
    return model, prec_30, hitrate, model_params


def fit_predict_als_model(df_train, df_test, **params):
    K = params['k']
    
    df_train2 = df_train.copy()
    df_train2['interaction'] = 1
    user_item_matrix = coo_matrix(
        (df_train2['interaction'].values, (df_train2['user_index'].values, df_train2['node_index'].values))
    ).tocsr()

    test_users = df_test['user_index'].unique()

    model = implicit.als.AlternatingLeastSquares(
        factors=params['edim'],
        regularization=params['regularization'],
        iterations=params['epoch'],
        random_state=SEED
    )
    model.fit(user_item_matrix)

    preds = []
    for user in test_users:
        scores = model.recommend(user, user_item_matrix[user], N=K, filter_already_liked_items=False)
        preds.append(scores[0])
    
    df_preds = pd.DataFrame({
        'node_index': np.concatenate(preds),
        'user_index': np.repeat(test_users, K),
        'rank': np.tile(np.arange(K), len(test_users))
    })

    df_preds = df_preds.explode(['node_index', 'rank']).merge(
        df_test[['user_index', 'node_index']].assign(relevant=1).drop_duplicates(),
        on=['user_index', 'node_index'],
        how='left'
    )
    df_preds['relevant'] = df_preds['relevant'].fillna(0)
    
    prec_30 = calc_prec(df_preds, K)
    hitrate = calc_hitrate(df_preds, K)
    
    model_keys = ['regularization', 'edim', 'epoch']
    model_params = {key: params[key] for key in params if key in model_keys}
    
    return model, prec_30, hitrate, model_params


def fit_predict_lightfm_model(df_train, df_test, **params):
    K = params['k']
    
    # dataset = LFM_Dataset()
    # dataset.fit(
    #     users=df_train['user_index'].unique(),
    #     items=df_train['node_index'].unique()
    # )
    # train_interactions, _ = dataset.build_interactions(
    #     ((user, item) for user, item in zip(
    #         df_train['user_index'], df_train['node_index']
    #     ))
    # )
    # test_interactions, _ = dataset.build_interactions(
    #     ((user, item) for user, item in zip(
    #         df_test['user_index'], df_test['node_index']
    #     ))
    # )
    
    # model = LightFM(
    #     loss=params['loss'],
    #     max_sampled=params['num_negatives'],
    #     no_components=params['edim'],
    #     learning_rate=params['lr'],
    #     random_state=SEED
    # )
    # model.fit(train_interactions, epochs=params['epoch'])
    
    # prec_30 = precision_at_k(model, test_interactions, k=K).mean()
    # hitrate = recall_at_k(model, test_interactions, k=K).mean()
    
    # model_keys = ['num_negatives', 'edim', 'epoch', 'lr', 'loss']
    # model_params = {key: params[key] for key in params if key in model_keys}
    
    # return model, prec_30, hitrate, model_params


def main(model_name: str, run_name: str, **params):

    with open('data/node2name.json', 'r') as f:
        node2name = json.load(f)

    node2name = {int(k):v for k,v in node2name.items()}
    
    df = pd.read_parquet('data/clickstream.parque')
    df = df.head(100_000)
    df['is_train'] = df['event_date'] < df['event_date'].max() - pd.Timedelta('2 day')
    df['names'] = df['node_id'].map(node2name)
    train_cooks = df[df['is_train']]['cookie_id'].unique()
    train_items = df[df['is_train']]['node_id'].unique()

    df = df[(df['cookie_id'].isin(train_cooks)) & (df['node_id'].isin(train_items))]
    user_indes, _ = pd.factorize(df['cookie_id'])
    df['user_index'] = user_indes

    node_indes, _ = pd.factorize(df['node_id'])
    df['node_index'] = node_indes
    df_train, df_test = df[df['is_train']], df[~df['is_train']]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    

    if model_name == 'LatentFactorModel':
        model, prec_30, hitrate, model_params = fit_predict_latent_factory_model(
            df, df_train, df_test,**params
        )

    if model_name == 'AlternatingLeastSquares':
        model, prec_30, hitrate, model_params = fit_predict_als_model(
            df_train, df_test, **params
        )
        
    if model_name == 'LightFM':
        model, prec_30, hitrate, model_params = fit_predict_lightfm_model(
            df_train, df_test, **params
        )

    mlflow.set_tracking_uri('http://84.201.128.89:90/')

    mlflow.set_experiment('homework-pipeline-yvmazepa')

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(
            {'precision_30': prec_30, 'hitrate_30': hitrate}
        )
        mlflow.log_params(
            {
                'model_name': model_name,
            } | model_params
        )

        # mlflow.sklearn.log_model(model, 'model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='model name', default = 'LatentFactorModel')
    parser.add_argument('--run-name', help='run name', default = 'LatentFactorModel_base')
    parser.add_argument('--batch-size', type=int, help='batch size', default=50000)
    parser.add_argument('--num-negatives', type=int, help='num negatives', default=5)
    parser.add_argument('--edim', type=int, help='edim', default=128)
    parser.add_argument('--epoch', type=int, help='epoch', default=10)
    parser.add_argument('--k', type=int, help='k', default=30)
    parser.add_argument('--lr', type=float, help='lr', default=1.0)
    parser.add_argument('--optimizer', help='optimizer', default='Adam')
    parser.add_argument('--regularization', type=float, help='regularization', default=0.1)
    parser.add_argument('--loss', help='loss', default='warp')

    args = parser.parse_args()
    # main(args.model_name,  run_name = args.run_name)
    params = args.__dict__
    main(**params)
