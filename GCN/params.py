import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import pickle
import requests
import torch
import random

def pload(path):
    # Load a pickle file
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res

def pstore(x, path):
    # Store an object in a pickle file
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))

def create_combined_text(filtered_stats):
    # 필요한 컬럼들 추출 및 결합
    combined_text = "\n".join(
        f"Champion: {row['Champion']}, WinRate: {row['WinRate']}, PickRate: {row['PickRate']}"
        for _, row in filtered_stats.iterrows()
    )
    return combined_text

def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero=False):
    print('Computing \Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)  # Item-Item interaction matrix
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors), dtype=torch.long)
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis=0).reshape(-1)
    users_D = np.sum(A, axis=1).reshape(-1)

    # Avoid division by zero
    users_D[users_D == 0] = 1
    items_D[items_D == 0] = 1

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \Omega OK!')
    return res_mat, res_sim_mat.float()


def save_train_text_dict(train_text_dict, file_path):
    # 딕셔너리를 파일로 저장
    with open(file_path, 'wb') as f:
        pickle.dump(train_text_dict, f)
    print(f"Dictionary saved to {file_path}")

def load_train_text_dict(file_path):
    # 딕셔너리를 파일에서 불러오기
    with open(file_path, 'rb') as f:
        train_text_dict = pickle.load(f)
    print(f"Dictionary loaded from {file_path}")
    return train_text_dict

def train_text_dict_exists(file_path):
    # 파일이 존재하는지 검사
    exists = os.path.exists(file_path)
    if exists:
        print(f"File {file_path} exists.")
    else:
        print(f"File {file_path} does not exist.")
    return exists


def load_data(csv_file_path,winrate_mode = True,testcutting=3000):
    # Load train data from the user preference CSV file
    user_preference_stats = pd.read_csv(csv_file_path)
    champion_stats = pd.read_csv('../dataprocessing/id_champion_stats.csv')

    # Extract user IDs and champion preferences
    user_ids = user_preference_stats['ID'].values
    winrate_sorted_champions = user_preference_stats['WinRate_Sorted_Champions'].apply(lambda x: eval(x)).values
    pickrate_sorted_champions = user_preference_stats['PickRate_Sorted_Champions'].apply(lambda x: eval(x)).values

    if winrate_mode :
        chosen = winrate_sorted_champions

    else:
        chosen = pickrate_sorted_champions

    # Fetch latest champion data for labeling
    version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    versions = requests.get(version_url).json()
    latest_version = versions[0]
    champion_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
    response = requests.get(champion_url)
    champion_data = response.json()
    champion_name_to_key = {champion['name']: int(champion['key']) for champion in champion_data['data'].values()}
    champion_keys = sorted([int(champion['key']) for champion in champion_data['data'].values()])
    champion_key_to_index = {key: idx for idx, key in enumerate(champion_keys)}  # Map champion key to fixed index (0 to 167)
    index_to_champion_name = {champion_key_to_index[key]:name for name, key in champion_name_to_key.items()}

    print(champion_key_to_index)
    # Map user IDs to indices
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    #print(len(user_id_to_index))
    train_data = []
    train_text_dict = {}  # Dictionary to store text data for easy access
    m_item = len(champion_key_to_index)  # Number of unique champions
    interacted_items = [[] for _ in range(len(user_id_to_index))]

    exist_check = train_text_dict_exists('../train_text_dict.pkl')
    if exist_check:
        train_text_dict = load_train_text_dict('../train_text_dict.pkl')

    # Construct train data from the CSV contents
    for idx, user_id in enumerate(user_ids):
        user_index = user_id_to_index[user_id]  # Convert user ID to index
        if not exist_check:
           # txt = create_combined_text(champion_stats[champion_stats['ID'] == user_id])
            train_text_dict[user_index] = champion_stats[champion_stats['ID'] == user_id]  # Store text data in dictionary

        for champion in chosen[idx]:
            champion_key = champion_name_to_key.get(champion, None)
            if champion_key is not None:
                champion_index = champion_key_to_index[champion_key]  # Convert champion key to fixed index (0 to 167)
                train_data.append([user_index, champion_index])

    save_train_text_dict(train_text_dict, '../train_text_dict.pkl')

    random.seed(42)
    test_data_indices = random.sample(range(len(train_data)), testcutting)
    test_data = [train_data[i] for i in test_data_indices]
    train_data = [train_data[i] for i in range(len(train_data)) if i not in test_data_indices]

    mask = torch.zeros(len(user_id_to_index), len(index_to_champion_name))
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    train_data = np.array(train_data)

    # Create train matrix
    n_user = len(user_id_to_index)  # Number of unique users
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_ground_truth_list = [[] for _ in range(len(user_id_to_index))]
    user_set = set()

    # test_data에서 사용자 ID를 집합에 추가
    for (u, i) in test_data:
        user_set.add(u)
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)
    print(len(user_set), "유저")

    # Construct degree matrix for graphmf
    items_D = np.sum(train_mat, axis=0).reshape(-1)
    users_D = np.sum(train_mat, axis=1).reshape(-1)

    # Avoid division by zero
    users_D[users_D == 0] = 1
    items_D[items_D == 0] = 1

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    beta_UD = torch.from_numpy(beta_uD).reshape(-1)
    beta_iD = torch.from_numpy(beta_iD).reshape(-1)

    return train_data,test_data, train_mat, n_user, m_item, beta_UD, beta_iD, index_to_champion_name, interacted_items, train_text_dict,mask,test_ground_truth_list

# Example usage
# csv_file_path = '../user_winrate_pickrate_sorted_champions.csv'
# train_data, train_mat, n_user, m_item, constraint_mat, index_to_champion_name, train_text_dict = load_data(csv_file_path)
# ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, 10)

# Replace ii_neighbor_mat indices with champion names
# try:
#     ii_neighbor_names = [[index_to_champion_name[int(idx)] for idx in row] for row in ii_neighbor_mat]
# except KeyError as e:
#     print(f"KeyError: {e}. Please check if the champion index is valid.")

# for ii in ii_neighbor_names:
#     print(ii)
# print(ii_neighbor_mat)
# print(ii_constraint_mat)
# print(constraint_mat)
