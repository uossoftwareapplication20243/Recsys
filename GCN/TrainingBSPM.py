import torch.utils.data as data
from torch.amp import autocast
import time
from torchdiffeq import odeint
from eval_code import *
from params import *
import wandb
from sparsesvd import sparsesvd


class BSPM(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat

        self.idl_solver = 'euler'
        self.blur_solver = 'euler'
        self.sharpen_solver = 'rk4'

        self.idl_beta = 0.2
        self.factor_dim = 448

        idl_T = 1
        idl_K = 1

        blur_T = 1
        blur_K = 1

        sharpen_T = 2.5
        sharpen_K = 1

        self.device = 'cuda'
        self.idl_times = torch.linspace(0, idl_T, idl_K + 1).float().to(self.device)
        self.blurring_times = torch.linspace(0, blur_T, blur_K + 1).float().to(self.device)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K + 1).float().to(self.device)

        self.final_sharpening = True

    def train(self):
        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5, where=rowsum != 0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5, where=colsum != 0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        self.d_mat_i = d_mat

        self.d_mat_i_inv = sp.diags(1 / d_inv, 0)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        del norm_adj, d_mat

        linear_Filter = self.norm_adj.T @ self.norm_adj
        self.linear_Filter = self.convert_sp_mat_to_sp_tensor(linear_Filter).to_dense().to(self.device)



    def sharpenFunction(self, t, r):
        out = r @ self.linear_Filter
        return -out


    def getUsersRating(self, batch_users):
        batch_test = batch_users.to_sparse()

        with torch.no_grad():

            blurred_out = torch.mm(batch_test.to_dense(), self.linear_Filter)
            if self.final_sharpening == True:  # EM
                sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)

        U_2 = sharpened_out[-1]
        ret = U_2
        return ret

    def handle_cold_start_user(self, selected_items, top_k=5):
        with torch.no_grad():
            blurred_out = torch.exp(-torch.eye(self.linear_Filter.size(0), device=self.device) @ self.linear_Filter)

            sharpening_times = torch.linspace(0, 1, 50).to(self.device)  # 샤프닝 시간 설정

            sharpened_out = odeint(
                func=lambda t, r: -r @ self.linear_Filter,
                y0=blurred_out,
                t=sharpening_times,
                method='rk4'
            )

        with torch.no_grad():
            selected_items_vector = torch.zeros((1, self.adj_mat.shape[1]), device=self.device)
            selected_items_vector[0, selected_items] = 1

            item_scores = torch.mm(selected_items_vector, sharpened_out[-1])
            _, top_items = torch.topk(item_scores.squeeze(), k=top_k)
        return top_items

    def convert_sp_mat_to_sp_tensor(self, X):
        # X가 이미 PyTorch Tensor라면 변환 작업 필요 없음
        if isinstance(X, torch.Tensor):
            if X.is_sparse:
                return X  # 이미 희소 텐서라면 반환
            else:
                raise ValueError("Input is a dense tensor, expected a sparse matrix.")

        # X가 SciPy 희소 행렬인지 확인
        if sp.issparse(X):
            coo = X.tocoo().astype(np.float32)
            row = torch.Tensor(coo.row).long()
            col = torch.Tensor(coo.col).long()
            index = torch.stack([row, col])
            data = torch.FloatTensor(coo.data)
            return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

        raise TypeError("Input must be a PyTorch sparse tensor or a SciPy sparse matrix.")


def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue, r, k)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def train(model,adj_mat, test_loader, mask, test_ground_truth_list, ):
    model.train()                                                                                        #win : 2281, pick: 1710
    F1_score, Precision, Recall, NDCG = test(model, adj_mat,test_loader, test_ground_truth_list, mask, 10,2281) #이 부분 갈아끼워야함.



    print(" F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(F1_score,
                                                                                                    Precision,
                                                                                                    Recall,
                                                                                                NDCG))
    return model,F1_score, Precision, Recall, NDCG



def test(model,adj_mat, test_loader, test_ground_truth_list, mask, topk, n_user):
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        for idx, batch_users in enumerate(test_loader):

            if not isinstance(adj_mat, torch.Tensor):
                adj_mat = model.convert_sp_mat_to_sp_tensor(adj_mat)

            # 밀집 텐서로 변환
            adj_mat = adj_mat.to_dense()
            batch_ratings = adj_mat[batch_users[0], :].to('cuda')

            rating = model.getUsersRating(batch_ratings).cpu()

            rating += mask[batch_users[0]]

            _, rating_K = torch.topk(rating, k=topk)

            rating_list.append(rating_K)
            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users[0]])


    selected_indices = torch.randperm(169)[:3]
    print(selected_indices)
    print(model.handle_cold_start_user(selected_indices))

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg

    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


if __name__ == "__main__":
    csv_file_path = '../dataprocessing/user_winrate_pickrate_sorted_champions.csv'

    # Load winrate data
    (train_data_win, test_data_win, train_mat_win, n_user_win, m_item_win, beta_UD_win, beta_iD_win,
     index_to_champion_name_win, interacted_items_win, train_text_dict_win, mask_win,
     test_ground_truth_list_win) = load_data(csv_file_path, winrate_mode=True,testcutting=3000)

  #  np.save('./train_mat_win.npy', np.array(train_mat_win.todense()))
    print(train_mat_win.shape)
    test_loader_win = data.DataLoader(test_data_win, batch_size=32, shuffle=False, num_workers=22)  # usernum


    # Load pickrate data
    (train_data_pick, test_data_pick, train_mat_pick, n_user_pick, m_item_pick, beta_UD_pick, beta_iD_pick,
    index_to_champion_name_pick, interacted_items_pick, train_text_dict_pick, mask_pick,
     test_ground_truth_list_pick) = load_data(csv_file_path, winrate_mode=False,testcutting=2000)

    test_loader_pick = data.DataLoader(test_data_pick, batch_size=1024, shuffle=False, num_workers=22)  # usernum
   # np.save('./train_mat_pick.npy', np.array(train_mat_pick.todense()))

    # Train UltraGCN models
  #  wandb.login(key='9d262c1061921ff5bfbd6709191225fcf71fece0')

    # wandb 초기화
   # wandb.init(project='Ultragcn-training_test6')
   # wandb.run.name = "train with part data winrate"
    #wandb.run.save()

    best_ndcg_win = 0
    best_ndcg_pick = 0
    best_model_win = None
    best_model_pick = None

    directory = '../saving_model'

    ultragcn,F1_score_win, Precision_win, Recall_win, NDCG_win  =\
         train( BSPM(train_mat_win),train_mat_win, test_loader_win, mask_win, test_ground_truth_list_win)

   # ultragcn, F1_score_pick, Precision_pick, Recall_pick, NDCG_pick = \
    #    train( BSPM(train_mat_pick),train_mat_pick, test_loader_pick, mask_pick, test_ground_truth_list_pick)

    # if NDCG_win > best_ndcg_win:
    #  best_ndcg_win = NDCG_win
    # best_model_win = ultragcn.state_dict()
    # torch.save(best_model_win, os.path.join(directory, 'part_best_model_win.pth'))


    # wandb에 메트릭 기록
    #wandb.log({
        # 'F1 Score (WinRate)': F1_score_win,
       #  'Precision (WinRate)': Precision_win,
      #   'Recall (WinRate)': Recall_win,
     #    'NDCG (WinRate)': NDCG_win,
     #   'F1 Score (PickRate)': F1_score_pick,
      #  'Precision (PickRate)': Precision_pick,
        #'Recall (PickRate)': Recall_pick,
        #'NDCG (PickRate)': NDCG_pick
    #})
    #wandb.finish()
