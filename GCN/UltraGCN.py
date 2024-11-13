import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda.amp import autocast, GradScaler
import os
import gc
import time
import argparse
#from GCN import *
from .eval_code import *
from .sampling import *
from .params import *
import wandb
from transformers import DistilBertTokenizer, DistilBertModel


class UltraGCNWithDistilBERT(nn.Module):
    def __init__(self, beta_UD, beta_iD, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCNWithDistilBERT, self).__init__()
        self.item_num = 169
        self.embedding_dim = 768
        self.w1 = 1e-8
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1e-8

        self.negative_weight = 500
        self.gamma = 1e-4
        self.lambda_ = 2.75

        # DistilBERT 모델과 토크나이저 초기화
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        device = 'cuda'

        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim).to(device)

        if beta_UD is not None:
            self.beta_UD = beta_UD.to(device)
            self.beta_iD = beta_iD.to(device)
            self.ii_constraint_mat = ii_constraint_mat.to(device)
            self.ii_neighbor_mat = ii_neighbor_mat.to(device)

        self.initial_weight = 1e-4
        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_user_embeddings(self, user_texts):
        device = self.get_device()
        tokens = self.tokenizer(user_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        user_embeddings = self.bert_model(**tokens).last_hidden_state[:, 0, :]
        return user_embeddings

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.beta_UD[users],  self.beta_iD[pos_items])

            pos_weight = self.w1 + self.w2 * pos_weight.to(device)
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)

        if self.w4 > 0:
            # users와 neg_items를 동일한 장치로 이동
            beta_UD_users = self.beta_UD[users].to(device)
            neg_items_flat = neg_items.flatten().cpu()
            beta_iD_neg_items = self.beta_iD[neg_items_flat]

            # 필요 없는 CPU/GPU 전환을 최소화한 형태로 수정
            neg_weight = torch.mul(
                torch.repeat_interleave(beta_UD_users, neg_items.size(1)),
                beta_iD_neg_items.to(device)
            )

            # 혹은 필요 시 to(device)를 붙여 한 번 더 처리
            neg_weight = neg_weight.to(device)

            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)

        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def cal_loss_L(self,  user_texts, pos_items, neg_items, omega_weight):
        device = self.get_device()

        user_embeds = self.get_user_embeddings(user_texts).to(device)
        pos_embeds = self.item_embeds(pos_items.reshape(-1,1).to(device))
        neg_embeds = self.item_embeds(neg_items.to(device))

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels,
                                                      weight=omega_weight[len(pos_scores):].view(neg_scores.size()),
                                                      reduction='none').mean(dim=-1)

        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)],
                                                      reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))
        sim_scores = self.ii_constraint_mat[pos_items].to(device)

        user_embeds = self.get_user_embeddings(users).unsqueeze(1)
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, user, user_texts, pos_items, neg_items):
        omega_weight = self.get_omegas(user, pos_items, neg_items)

        loss = self.cal_loss_L(user_texts, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()

        loss += self.lambda_ * self.cal_loss_I(user_texts, pos_items)
        return loss

    def test_foward(self, user_texts):
        device = self.get_device()
        items = torch.arange(self.item_num).to(device)
        user_embeds = self.get_user_embeddings(user_texts).to(device)
        item_embeds = self.item_embeds(items)

        return user_embeds.mm(item_embeds.t())

    def get_device(self):
        return self.item_embeds.weight.device


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

def train(model, optimizer, train_loader,interacted_items,train_text_dict, test_loader, mask, test_ground_truth_list, index_to_champion_name,mode):

    scaler = GradScaler()
    model.train()
    start_time = time.time()
    for batch, x  in enumerate(train_loader):
        users, pos_items, neg_items, text = Sampling(x, 169, 160, interacted_items, train_text_dict,index_to_champion_name, True,mode)

        model.zero_grad()
        with autocast():
            loss = model(users, text, pos_items, neg_items.cuda())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))

    start_time = time.time()  # topk   #usernum
    F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, train_text_dict, mask, 10,600) #이 부분 갈아끼워야함.
    test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))


    print("train_time",train_time,"\n test_time",test_time)

    print("Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(loss.item(),
                                                                                                    F1_score,
                                                                                                    Precision,
                                                                                                    Recall,
                                                                                                NDCG))
    return model,F1_score, Precision, Recall, NDCG



def test(model, test_loader, test_ground_truth_list,train_text_dict, mask, topk, n_user):
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(test_loader):
            user_texts = [create_combined_text(train_text_dict[user.item()]) for user in batch_users[0]]  # 임시로 사용자 텍스트 추가
            with autocast():
                rating = model.test_foward( user_texts).cpu()

            rating += mask[batch_users[0]]

            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)
            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users[0]])

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
    #(train_data_win, test_data_win, train_mat_win, n_user_win, m_item_win, beta_UD_win, beta_iD_win,
     #index_to_champion_name_win, interacted_items_win, train_text_dict_win, mask_win,
     #test_ground_truth_list_win) = load_data(csv_file_path, winrate_mode=True,testcutting=3000)

    #train_loader_win = data.DataLoader(train_data_win, batch_size=512, shuffle=True, num_workers=22)
    #test_loader_win = data.DataLoader(test_data_win, batch_size=32, shuffle=False, num_workers=22)  # usernum

    # Generate ii_constraint matrices for winrate data
    #ii_neighbor_mat_win, ii_constraint_mat_win = get_ii_constraint_mat(train_mat_win, 10)  # ii_neighbor_num

    # Load pickrate data
    (train_data_pick, test_data_pick, train_mat_pick, n_user_pick, m_item_pick, beta_UD_pick, beta_iD_pick,
     index_to_champion_name_pick, interacted_items_pick, train_text_dict_pick, mask_pick,
     test_ground_truth_list_pick) = load_data(csv_file_path, winrate_mode=False,testcutting=2000)

    train_loader_pick = data.DataLoader(train_data_pick, batch_size=512, shuffle=True, num_workers=22)
    test_loader_pick = data.DataLoader(test_data_pick, batch_size=32, shuffle=False, num_workers=22)  # usernum

    ii_neighbor_mat_pick, ii_constraint_mat_pick = get_ii_constraint_mat(train_mat_pick, 10)  # ii_neighbor_num

    # Create UltraGCN model
    #ultragcn = UltraGCNWithDistilBERT(beta_UD_win, beta_iD_win, ii_constraint_mat_win.cuda(),ii_neighbor_mat_win.cuda()).cuda()

    ultragcn = UltraGCNWithDistilBERT(beta_UD_pick, beta_iD_pick, ii_constraint_mat_pick.cuda(),
                                           ii_neighbor_mat_pick.cuda()).cuda()
    optimizer = torch.optim.AdamW(ultragcn.parameters(), lr=5e-4,weight_decay=1e-4)

    # Train UltraGCN models
    wandb.login(key='9d262c1061921ff5bfbd6709191225fcf71fece0')

    # wandb 초기화
    wandb.init(project='Ultragcn-training_test4')
    wandb.run.name = "train with part data pickrate"
    wandb.run.save()

    best_ndcg_win = 0
    best_ndcg_pick = 0
    best_model_win = None
    best_model_pick = None

    directory = '../saving_model'

    mode=1
    for iter in range(75):

     #   ultragcn,F1_score_win, Precision_win, Recall_win, NDCG_win  =\
      #      train(ultragcn, optimizer, train_loader_win, interacted_items_win,
       #           train_text_dict_win, test_loader_win, mask_win, test_ground_truth_list_win,index_to_champion_name_win,mode)

        ultragcn,F1_score_pick, Precision_pick, Recall_pick, NDCG_pick  =\
            train(ultragcn, optimizer, train_loader_pick, interacted_items_pick, train_text_dict_pick, test_loader_pick,
                  mask_pick,test_ground_truth_list_pick,index_to_champion_name_pick,mode)

        #if NDCG_win > best_ndcg_win:
          #  best_ndcg_win = NDCG_win
         #   best_model_win = ultragcn.state_dict()
        #    torch.save(best_model_win, os.path.join(directory, 'part_best_model_win.pth'))

        # 최고의 모델 업데이트 및 저장 (PickRate 기준)
        if NDCG_pick > best_ndcg_pick:
            best_ndcg_pick = NDCG_pick
            best_model_pick = ultragcn.state_dict()
            torch.save(best_model_pick, os.path.join(directory, 'part_best_model_pick.pth'))


        # wandb에 메트릭 기록
        wandb.log({
            #'F1 Score (WinRate)': F1_score_win,
            #'Precision (WinRate)': Precision_win,
            #'Recall (WinRate)': Recall_win,
            #'NDCG (WinRate)': NDCG_win,
            'F1 Score (PickRate)': F1_score_pick,
            'Precision (PickRate)': Precision_pick,
            'Recall (PickRate)': Recall_pick,
            'NDCG (PickRate)': NDCG_pick
        })
    wandb.finish()
