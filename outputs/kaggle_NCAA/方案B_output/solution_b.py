#!/usr/bin/env python3
"""
方案B: 多视角融合 — 贝叶斯 + 神经嵌入 + 评分系统
NCAA March Madness 2026 概率预测

依赖: numpy, pandas, scipy, sklearn, torch
可选: pymc (如果启用完整MCMC, 默认用MAP加速)

Brier Score 目标: < 0.115
"""

import os
import sys
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# ──────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# 第0部分: 数据加载与预处理
# ==============================================================

class DataLoader:
    """加载并预处理 NCAA 数据"""

    def __init__(self, gender='M'):
        self.gender = gender  # 'M' 或 'W'
        self.prefix = gender
        self._load_all()

    def _load_all(self):
        # 球队信息
        teams = pd.read_csv(os.path.join(DATA_DIR, f'{self.prefix}Teams.csv'))
        self.teams_df = teams
        self.team_id_to_name = dict(zip(teams.TeamID, teams.TeamName))

        # 赛季信息
        self.seasons = pd.read_csv(os.path.join(DATA_DIR, f'{self.prefix}Seasons.csv'))

        # 种子信息
        seeds = pd.read_csv(os.path.join(DATA_DIR, f'{self.prefix}NCAATourneySeeds.csv'))
        seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)
        self.seeds = seeds

        # 常规赛(紧凑结果)
        reg = pd.read_csv(os.path.join(DATA_DIR, f'{self.prefix}RegularSeasonCompactResults.csv'))
        self.regular_season = reg

        # 常规赛(详细结果) — 用于四因素分析
        detail_path = os.path.join(DATA_DIR, f'{self.prefix}RegularSeasonDetailedResults.csv')
        if os.path.exists(detail_path):
            self.regular_season_detail = pd.read_csv(detail_path)
        else:
            self.regular_season_detail = None

        # 锦标赛(紧凑结果)
        tourney = pd.read_csv(os.path.join(DATA_DIR, f'{self.prefix}NCAATourneyCompactResults.csv'))
        self.tourney = tourney

        # 锦标赛详细结果
        tourney_detail_path = os.path.join(DATA_DIR, f'{self.prefix}NCAATourneyDetailedResults.csv')
        if os.path.exists(tourney_detail_path):
            self.tourney_detail = pd.read_csv(tourney_detail_path)
        else:
            self.tourney_detail = None

        # 锦标赛槽位(用于构建2026 bracket)
        slots = pd.read_csv(os.path.join(DATA_DIR, f'{self.prefix}NCAATourneySlots.csv'))
        self.slots = slots

        # 所有比赛合并
        all_games = self._merge_games(reg, tourney)
        self.all_games = all_games

        # 所有球队ID(构建索引)
        all_team_ids = sorted(teams.TeamID.unique())
        self.team_id_to_idx = {tid: i for i, tid in enumerate(all_team_ids)}
        self.team_idx_to_id = {i: tid for tid, i in self.team_id_to_idx.items()}
        self.n_teams = len(all_team_ids)

        # Massey Ordinals(仅男子)
        if self.gender == 'M':
            ordinals_path = os.path.join(DATA_DIR, 'MMasseyOrdinals.csv')
            if os.path.exists(ordinals_path):
                self.ordinals = pd.read_csv(ordinals_path)
            else:
                self.ordinals = None
        else:
            self.ordinals = None

    def _merge_games(self, reg, tourney):
        """合并常规赛和锦标赛，添加时间戳和权重信息"""
        for df in [reg, tourney]:
            df['is_tourney'] = 1 if df is tourney else 0
            df['game_id'] = df.index.values + 1 if df is reg else -(df.index.values + 1)

        merged = pd.concat([reg, tourney], ignore_index=True)
        merged['home_team'] = merged.apply(
            lambda r: r.WTeamID if r.WLoc == 'H' else (r.LTeamID if r.WLoc == 'A' else -1),
            axis=1
        )
        merged['neutral_site'] = (merged.WLoc == 'N').astype(int)
        return merged.sort_values(['Season', 'DayNum']).reset_index(drop=True)

    def get_team_idx(self, team_id):
        return self.team_id_to_idx.get(team_id, -1)

    def get_team_name(self, team_id):
        return self.team_id_to_dict.get(team_id, f'Team_{team_id}')

    def get_team_id(self, idx):
        return self.team_idx_to_id[idx]


# ==============================================================
# 第1部分: 视角1 — 贝叶斯层级模型 (MAP估计)
# ==============================================================

class BayesianStrengthModel:
    """
    基于球队潜在实力的贝叶斯层级模型
    模型: y_ij ~ Bernoulli(sigmoid(θ_i - θ_j + γ·home))
    先验: θ_i ~ Normal(μ, σ²), γ ~ Normal(0, 0.5)
    估计: MAP (scipy minimize) 或 MCMC (PyMC)
    """

    def __init__(self, time_decay_lambda=0.005, use_mcmc=False, season_carryover=0.85):
        self.time_decay_lambda = time_decay_lambda
        self.use_mcmc = use_mcmc
        self.season_carryover = season_carryover
        self.theta = None        # team strength params
        self.home_adv = 0.0      # home advantage coefficient
        self.mu = 0.0            # prior mean
        self.sigma = 1.0         # prior std
        self.n_teams = 0
        self.fitted = False

    def _compute_time_weights(self, seasons, day_nums, current_season=2026):
        """指数时间衰减权重"""
        days_since = (current_season - seasons) * 365 + (365 - day_nums)
        days_since = np.maximum(days_since, 0)
        weights = np.exp(-self.time_decay_lambda * days_since)
        return weights

    def fit(self, team_a_idx, team_b_idx, y, home_indicator,
            seasons=None, day_nums=None, current_season=2026,
            season_carryover_prior=None, n_teams_full=None):
        """
        MAP估计球队实力参数

        Parameters:
        -----------
        team_a_idx : array, 球队A的索引
        team_b_idx : array, 球队B的索引
        y : array, 比赛结果(1=A胜, 0=B胜)
        home_indicator : array, 主场标志(1=home, -1=away, 0=neutral)
        seasons : array, 赛季年份
        day_nums : array, 赛季内天数
        n_teams_full : int or None, 全局球队数(未在当前数据中出现的球队用theta=0)
        """
        if n_teams_full is not None:
            self.n_teams = n_teams_full
        else:
            self.n_teams = max(max(team_a_idx), max(team_b_idx)) + 1
        n = len(y)

        # 时间权重
        if seasons is not None and day_nums is not None:
            w = self._compute_time_weights(seasons, day_nums, current_season)
        else:
            w = np.ones(n)

        # 赛季间传递先验 (从carryover_prior继承)
        theta_prior_mean = np.zeros(self.n_teams)
        theta_prior_std = np.ones(self.n_teams) * 10.0  # 宽松先验

        if season_carryover_prior is not None:
            theta_prior_mean = np.array(season_carryover_prior) * self.season_carryover
            theta_prior_std = np.ones(self.n_teams) * 2.0

        def neg_log_posterior(params):
            theta = params[:self.n_teams]
            home = params[self.n_teams]

            # 对数似然 (加权)
            logit_p = theta[team_a_idx] - theta[team_b_idx] + home * home_indicator
            # 截断防止数值溢出
            logit_p = np.clip(logit_p, -20, 20)
            p = expit(logit_p)
            eps = 1e-10
            ll = np.sum(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

            # 对数先验: θ ~ Normal(θ_prior_mean, θ_prior_std)
            lp_theta = -0.5 * np.sum(((theta - theta_prior_mean) / theta_prior_std) ** 2)

            # 对数先验: home ~ Normal(0, 0.5)
            lp_home = -0.5 * (home / 0.5) ** 2

            return -(ll + lp_theta + lp_home)

        # 初始值
        x0 = np.zeros(self.n_teams + 1)

        result = minimize(
            neg_log_posterior,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-8}
        )

        self.theta = result.x[:self.n_teams]
        self.home_adv = result.x[self.n_teams]
        self.mu = float(np.mean(self.theta))
        self.sigma = float(np.std(self.theta))
        self.fitted = True

        return self

    def predict(self, team_a_idx, team_b_idx, home=0):
        """预测 P(team_a beats team_b)"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        logit_p = self.theta[team_a_idx] - self.theta[team_b_idx] + self.home_adv * home
        logit_p = np.clip(logit_p, -20, 20)
        return float(expit(logit_p))

    def predict_batch(self, team_a_idxs, team_b_idxs, home=0):
        logit_p = self.theta[team_a_idxs] - self.theta[team_b_idxs] + self.home_adv * home
        logit_p = np.clip(logit_p, -20, 20)
        return expit(logit_p)

    def get_theta(self):
        return self.theta.copy()


# ==============================================================
# 第2部分: 视角2 — 神经嵌入模型 (PyTorch)
# ==============================================================

class TeamEmbeddingNet(nn.Module):
    """球队嵌入神经网络"""

    def __init__(self, n_teams, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_teams, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, team_a, team_b):
        e_a = self.embedding(team_a)
        e_b = self.embedding(team_b)
        x = torch.cat([
            e_a, e_b,
            e_a - e_b,
            e_a * e_b,
            torch.abs(e_a - e_b)
        ], dim=1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class NeuralEmbeddingModel:
    """
    PyTorch 球队嵌入模型
    用嵌入式表示捕获球队风格和实力的联合表征
    """

    def __init__(self, n_teams, embed_dim=32, device='cpu'):
        self.n_teams = n_teams
        self.embed_dim = embed_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = TeamEmbeddingNet(n_teams, embed_dim).to(self.device)
        self.fitted = False
        self.train_losses = []

    def fit(self, team_a_idxs, team_b_idxs, y,
            n_epochs=50, batch_size=256, lr=0.001,
            weight_decay=1e-4, early_stop_patience=5,
            verbose=True):
        """
        训练神经嵌入模型

        Data augmentation: 每场比赛生成两个样本 (A赢B, B输A)
        """
        n = len(y)
        # Data augmentation: 每个比赛产生两个方向样本
        a_all = np.concatenate([team_a_idxs, team_b_idxs])
        b_all = np.concatenate([team_b_idxs, team_a_idxs])
        y_all = np.concatenate([y, 1 - y])

        dataset = TensorDataset(
            torch.LongTensor(a_all),
            torch.LongTensor(b_all),
            torch.FloatTensor(y_all)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5, min_lr=1e-5
        )

        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            self.model.train()
            epoch_losses = []
            for a, b, label in loader:
                a, b, label = a.to(self.device), b.to(self.device), label.to(self.device)
                pred = self.model(a, b)
                loss = F.binary_cross_entropy(pred, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [Embedding] Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"  [Embedding] Early stopping at epoch {epoch + 1}")
                    break

        # 恢复最佳状态
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.fitted = True
        return self

    def predict(self, team_a_idx, team_b_idx):
        """预测 P(team_a beats team_b)"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        with torch.no_grad():
            a = torch.LongTensor([team_a_idx]).to(self.device)
            b = torch.LongTensor([team_b_idx]).to(self.device)
            return float(self.model(a, b).cpu().numpy())

    def predict_batch(self, team_a_idxs, team_b_idxs):
        with torch.no_grad():
            a = torch.LongTensor(team_a_idxs).to(self.device)
            b = torch.LongTensor(team_b_idxs).to(self.device)
            return self.model(a, b).cpu().numpy()

    def get_embeddings(self):
        """返回嵌入矩阵 (n_teams x embed_dim) 用于可视化"""
        return self.model.embedding.weight.detach().cpu().numpy()


# ==============================================================
# 第3部分: 视角3 — 评分系统集成
# ==============================================================

class RatingSystemEnsemble:
    """
    多种评分系统的加权集成
    包含: Elo变体, Massey, Colley, PageRank, 效率评分, 四因素
    """

    def __init__(self):
        self.ratings = {}       # name → rating vector (n_teams,)
        self.weights = None     # logistic regression weights
        self.scales = {}        # name → scale for prob conversion
        self.n_teams = 0
        self.team_id_to_idx = {}
        self.sub_models = []    # (name, predict_fn, weight)
        self.fitted = False

    def _setup(self, n_teams, team_id_to_idx):
        self.n_teams = n_teams
        self.team_id_to_idx = team_id_to_idx

    def _elo(self, games_df, K=20, home_adv=100, season_reset=0.75,
             name='elo', adaptive=False):
        """
        计算Elo评分

        Parameters:
        -----------
        games_df : DataFrame with columns [Season, WTeamID, LTeamID, WScore, LScore, WLoc]
        K : Elo K-factor
        home_adv : 主场优势(ELO分)
        season_reset : 赛季重置比例
        """
        ratings = {}
        all_team_ids = set(games_df.WTeamID.unique()) | set(games_df.LTeamID.unique())

        # 按赛季排序
        games_sorted = games_df.sort_values(['Season', 'DayNum'])

        for _, game in games_sorted.iterrows():
            season = game['Season']
            winner = game['WTeamID']
            loser = game['LTeamID']
            w_score, l_score = game['WScore'], game['LScore']

            # 赛季重置
            for tid in all_team_ids:
                if tid not in ratings:
                    ratings[tid] = 1500.0

            # 主场调整
            if game['WLoc'] == 'H':
                home_team = winner
            elif game['WLoc'] == 'A':
                home_team = loser
            else:
                home_team = None

            # 计算预期胜率
            r_w = ratings[winner]
            r_l = ratings[loser]
            if home_team == winner:
                r_w += home_adv
            elif home_team == loser:
                r_l += home_adv

            e_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400.0))
            e_l = 1.0 - e_w

            # 计算净胜分调整
            if adaptive:
                margin = abs(w_score - l_score)
                mov_mult = np.log(max(margin, 1) + 1) / np.log(2)
                k_actual = K * mov_mult
            else:
                k_actual = K

            # 更新
            ratings[winner] += k_actual * (1 - e_w)
            ratings[loser] += k_actual * (0 - e_l)

        return np.array([ratings.get(tid, 1500.0)
                         for tid in sorted(self.team_id_to_idx.keys(), key=lambda x: self.team_id_to_idx[x])])

    def _massey(self, games_df):
        """
        Massey评分: 解线性系统 X'X r = X'y
        直觉: 通过净胜分矩阵求解球队相对实力
        """
        team_ids = sorted(self.team_id_to_idx.keys(), key=lambda x: self.team_id_to_idx[x])
        n = len(team_ids)
        tid_to_idx = {tid: i for i, tid in enumerate(team_ids)}

        X_rows = []
        y_vals = []

        for _, game in games_df.iterrows():
            i = tid_to_idx[game['WTeamID']]
            j = tid_to_idx[game['LTeamID']]
            margin = game['WScore'] - game['LScore']

            row = np.zeros(n)
            row[i] = 1
            row[j] = -1
            X_rows.append(row)
            y_vals.append(margin)

        X = np.array(X_rows)
        y = np.array(y_vals)

        # X'X + 小正则化(可逆性)
        XtX = X.T @ X + np.eye(n) * 1e-6
        Xty = X.T @ y

        # 约束: sum(r) = 0 (添加一行)
        XtX_aug = np.vstack([XtX, np.ones(n)])
        Xty_aug = np.append(Xty, 0)

        # 最小二乘
        try:
            r, _, _, _ = np.linalg.lstsq(XtX_aug, Xty_aug, rcond=None)
        except:
            r = np.zeros(n)

        return r[:n]

    def _colley(self, games_df):
        """
        Colley评分: 贝叶斯版本的Massey
        C_ii = 2 + n_i, C_ij = -n_ij, b_i = 1 + (w_i - l_i)/2
        """
        team_ids = sorted(self.team_id_to_idx.keys(), key=lambda x: self.team_id_to_idx[x])
        n = len(team_ids)
        tid_to_idx = {tid: i for i, tid in enumerate(team_ids)}

        C = np.eye(n) * 2
        b = np.ones(n)

        for _, game in games_df.iterrows():
            i = tid_to_idx[game['WTeamID']]
            j = tid_to_idx[game['LTeamID']]

            b[i] += 0.5
            b[j] -= 0.5

            C[i, j] -= 1
            C[j, i] -= 1
            C[i, i] += 1
            C[j, j] += 1

        try:
            r = np.linalg.solve(C, b)
        except:
            r = np.linalg.lstsq(C, b, rcond=None)[0]

        return r

    def _pagerank(self, games_df, damping=0.85, max_iter=100, tol=1e-8):
        """
        PageRank评分: 赢强队 → 高分
        有向图 A → B (A赢了B)
        """
        team_ids = sorted(self.team_id_to_idx.keys(), key=lambda x: self.team_id_to_idx[x])
        n = len(team_ids)
        tid_to_idx = {tid: i for i, tid in enumerate(team_ids)}

        # 出度(输给谁)
        out_degree = np.zeros(n)
        adj = np.zeros((n, n))

        for _, game in games_df.iterrows():
            winner = tid_to_idx[game['WTeamID']]
            loser = tid_to_idx[game['LTeamID']]
            adj[loser, winner] += 1  # loser → winner (输了指向赢家)
            out_degree[loser] += 1

        # 处理出度为0的节点
        out_degree[out_degree == 0] = 1

        # 转移矩阵
        P = adj / out_degree[:, np.newaxis]

        # Power iteration
        pr = np.ones(n) / n
        for _ in range(max_iter):
            pr_new = (1 - damping) / n + damping * P.T @ pr
            if np.linalg.norm(pr_new - pr) < tol:
                break
            pr = pr_new

        return pr

    def _efficiency_ratings(self, games_df):
        """
        进攻/防守效率评分(对手强度修正)
        OffRating = 场均得分(对手防守强度修正)
        DefRating = 场均失分(对手进攻强度修正)
        """
        team_ids = sorted(self.team_id_to_idx.keys(), key=lambda x: self.team_id_to_idx[x])
        n = len(team_ids)
        tid_to_idx = {tid: i for i, tid in enumerate(team_ids)}

        # 收集每场比赛的得分/失分
        scored = {tid: [] for tid in team_ids}
        conceded = {tid: [] for tid in team_ids}

        for _, game in games_df.iterrows():
            w, l = game['WTeamID'], game['LTeamID']
            ws, ls = game['WScore'], game['LScore']
            scored[w].append(ws)
            conceded[w].append(ls)
            scored[l].append(ls)
            conceded[l].append(ws)

        # 原始效率
        ortg = np.array([np.mean(scored[tid]) if scored[tid] else 100 for tid in team_ids])
        drtg = np.array([np.mean(conceded[tid]) if conceded[tid] else 100 for tid in team_ids])

        # 对手强度修正(迭代3轮)
        for _ in range(3):
            ortg_adj = np.zeros(n)
            drtg_adj = np.zeros(n)
            for idx, tid in enumerate(team_ids):
                opp_ortg = []
                opp_drtg = []
                for _, game in games_df.iterrows():
                    if game['WTeamID'] == tid:
                        opp_idx = tid_to_idx[game['LTeamID']]
                        opp_drtg.append(drtg[opp_idx])
                    elif game['LTeamID'] == tid:
                        opp_idx = tid_to_idx[game['WTeamID']]
                        opp_drtg.append(drtg[opp_idx])
                if opp_drtg:
                    ortg_adj[idx] = ortg[idx] - np.mean(opp_drtg) + np.mean(drtg)
                else:
                    ortg_adj[idx] = ortg[idx]
            ortg, drtg = ortg_adj, drtg_adj

        net_rating = ortg - drtg
        return net_rating

    def _four_factors(self, games_detail):
        """
        四因素分析
        需要详细比赛数据(MRegularSeasonDetailedResults.csv)
        """
        team_ids = sorted(self.team_id_to_idx.keys(), key=lambda x: self.team_id_to_idx[x])
        n = len(team_ids)
        tid_to_idx = {tid: i for i, tid in enumerate(team_ids)}

        # 累积统计
        stats = {tid: {
            'fgm': 0, 'fga': 0, 'fgm3': 0, 'fga3': 0,
            'ftm': 0, 'fta': 0,
            'or': 0, 'dr': 0, 'dr_opp': 0,
            'to': 0, 'poss': 0,
            'opp_fgm': 0, 'opp_fga': 0, 'opp_fgm3': 0, 'opp_ftm': 0, 'opp_fta': 0,
            'opp_or': 0, 'opp_dr': 0, 'opp_to': 0,
            'games': 0
        } for tid in team_ids}

        # 需要从详细数据中提取
        if games_detail is not None and len(games_detail) > 0:
            cols = games_detail.columns.tolist()
            for _, game in games_detail.iterrows():
                w, l = game['WTeamID'], game['LTeamID']
                if w in stats and l in stats:
                    # 胜方
                    stats[w]['fgm'] += game.get('WFGM', 0)
                    stats[w]['fga'] += game.get('WFGA', 0)
                    stats[w]['fgm3'] += game.get('WFGM3', 0)
                    stats[w]['ftm'] += game.get('WFTM', 0)
                    stats[w]['fta'] += game.get('WFTA', 0)
                    stats[w]['or'] += game.get('WOR', 0)
                    stats[w]['dr'] += game.get('WDR', 0)
                    stats[w]['to'] += game.get('WTO', 0)
                    stats[w]['games'] += 1

                    # 负方数据(对方视角)
                    stats[w]['opp_fgm'] += game.get('LFGM', 0)
                    stats[w]['opp_fga'] += game.get('LFGA', 0)
                    stats[w]['opp_fgm3'] += game.get('LFGM3', 0)
                    stats[w]['opp_ftm'] += game.get('LFTM', 0)
                    stats[w]['opp_fta'] += game.get('LFTA', 0)
                    stats[w]['opp_or'] += game.get('LOR', 0)
                    stats[w]['opp_dr'] += game.get('LDR', 0)
                    stats[w]['opp_to'] += game.get('LTO', 0)

                    # 负方
                    stats[l]['opp_fgm'] += game.get('WFGM', 0)
                    stats[l]['opp_fga'] += game.get('WFGA', 0)
                    stats[l]['opp_fgm3'] += game.get('WFGM3', 0)
                    stats[l]['opp_ftm'] += game.get('WFTM', 0)
                    stats[l]['opp_fta'] += game.get('WFTA', 0)
                    stats[l]['opp_or'] += game.get('WOR', 0)
                    stats[l]['opp_dr'] += game.get('WDR', 0)
                    stats[l]['opp_to'] += game.get('WTO', 0)
                    stats[l]['fgm'] += game.get('LFGM', 0)
                    stats[l]['fga'] += game.get('LFGA', 0)
                    stats[l]['fgm3'] += game.get('LFGM3', 0)
                    stats[l]['ftm'] += game.get('LFTM', 0)
                    stats[l]['fta'] += game.get('LFTA', 0)
                    stats[l]['or'] += game.get('LOR', 0)
                    stats[l]['dr'] += game.get('LDR', 0)
                    stats[l]['to'] += game.get('LTO', 0)
                    stats[l]['games'] += 1

        # 计算四因素(按场均)
        scores = np.zeros(n)
        for idx, tid in enumerate(team_ids):
            s = stats[tid]
            g = max(s['games'], 1)
            # eFG%
            efg = (s['fgm'] + 0.5 * s['fgm3']) / max(s['fga'], 1)
            # TO率
            to_rate = s['to'] / max(s['poss'] or s['fga'] + 0.44 * s['fta'] + s['to'], 1)
            # 篮板率
            reb_pct = s['or'] / max(s['or'] + s['dr_opp'], 1)
            # 罚球率
            ft_pct = s['ftm'] / max(s['fta'], 1)

            # 对方对应值
            opp_efg = (s['opp_fgm'] + 0.5 * s['opp_fgm3']) / max(s['opp_fga'], 1)
            opp_to_rate = s['opp_to'] / max(s['fga'] + 0.44 * s['fta'] + s['to'], 1)
            opp_reb_pct = s['opp_or'] / max(s['opp_or'] + s['dr'], 1)

            # 综合评分(权重: 40/25/20/15)
            scores[idx] = (
                0.40 * (efg - opp_efg) / 0.05 +
                0.25 * (opp_to_rate - to_rate) / 0.03 +
                0.20 * (reb_pct - opp_reb_pct) / 0.05 +
                0.15 * (ft_pct - 0.7) / 0.1
            )

        return scores

    def compute_all_ratings(self, regular_season, tourney,
                            regular_season_detail=None, tourney_detail=None,
                            current_season=2026):
        """计算所有评分系统"""
        # 合并常规赛+锦标赛(用于评分)
        combined = pd.concat([regular_season, tourney], ignore_index=True)

        # 只使用最近N年的数据
        recent = combined[combined.Season >= current_season - 5].copy()

        # 对每个赛季单独计算评分
        ratings_dict = {}

        # 1) Elo 变体
        ratings_dict['elo_A'] = self._elo(recent, K=20, home_adv=100, name='elo_A')
        ratings_dict['elo_B'] = self._elo(recent, K=30, home_adv=80, name='elo_B')
        ratings_dict['elo_C'] = self._elo(recent, K=20, home_adv=0, name='elo_C')
        ratings_dict['elo_D'] = self._elo(recent, K=20, home_adv=100, adaptive=True, name='elo_D')

        # 2) Massey
        ratings_dict['massey'] = self._massey(recent)

        # 3) Colley
        ratings_dict['colley'] = self._colley(recent)

        # 4) PageRank
        ratings_dict['pagerank'] = self._pagerank(recent)

        # 5) 效率评分
        ratings_dict['efficiency'] = self._efficiency_ratings(recent)

        # 6) 四因素法(需要详细比赛数据)
        detail_df = pd.concat([
            regular_season_detail if regular_season_detail is not None else pd.DataFrame(),
            tourney_detail if tourney_detail is not None else pd.DataFrame()
        ], ignore_index=True) if regular_season_detail is not None or tourney_detail is not None else None

        if detail_df is not None and len(detail_df) > 0:
            try:
                ratings_dict['four_factors'] = self._four_factors(detail_df)
            except:
                ratings_dict['four_factors'] = np.zeros(self.n_teams)

        self.ratings = ratings_dict
        return ratings_dict

    def fit_weights(self, team_a_idxs, team_b_idxs, y):
        """
        用逻辑回归学习各评分系统的权重
        数据增广: 每场比赛生成两个方向样本

        Features: 每个评分系统给出 P(A beats B) = sigmoid(score_diff)
        """
        n_systems = len(self.ratings)
        if n_systems == 0:
            raise ValueError("No ratings computed yet")

        system_names = list(self.ratings.keys())

        # 数据增广: 每场比赛产生两个样本(A赢B, B赢A)
        a_all = np.concatenate([team_a_idxs, team_b_idxs])
        b_all = np.concatenate([team_b_idxs, team_a_idxs])
        y_all = np.concatenate([y, 1 - y])

        X = np.zeros((len(y_all), n_systems))
        for j, name in enumerate(system_names):
            rating = self.ratings[name]
            score_diff = rating[a_all] - rating[b_all]
            scale = np.std(score_diff) + 1e-6
            self.scales[name] = scale
            X[:, j] = expit(score_diff / scale)

        # 逻辑回归(带L2正则) — 使用增广后的数据
        meta = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
        meta.fit(X, y_all)

        self.weights = meta.coef_[0]
        self.intercept = meta.intercept_[0]
        self.system_names = system_names
        self.fitted = True

        # 打印各系统权重
        print("\n  [Rating Systems] Learned weights:")
        for name, w in sorted(zip(system_names, self.weights),
                              key=lambda x: -abs(x[1])):
            print(f"    {name:20s}: {w:+.4f}")

        return self

    def predict(self, team_a_idx, team_b_idx):
        """用加权评分系统对预测"""
        features = []
        for name in self.system_names:
            diff = self.ratings[name][team_a_idx] - self.ratings[name][team_b_idx]
            p = expit(diff / self.scales[name])
            features.append(p)

        logit_p = np.dot(features, self.weights) + self.intercept
        return float(expit(logit_p))

    def predict_batch(self, team_a_idxs, team_b_idxs):
        features = np.zeros((len(team_a_idxs), len(self.system_names)))
        for j, name in enumerate(self.system_names):
            diff = self.ratings[name][team_a_idxs] - self.ratings[name][team_b_idxs]
            features[:, j] = expit(diff / self.scales[name])

        logit_p = features @ self.weights + self.intercept
        return expit(logit_p)


# ==============================================================
# 第4部分: 融合层
# ==============================================================

class FusionModel:
    """
    约束加权平均融合
    对三个视角的概率预测做加权平均，权重非负且和为1
    用 scipy 直接优化 Brier Score
    """

    def __init__(self):
        self.weights = None  # [w_bayes, w_nn, w_rating]
        self.fitted = False

    def fit(self, bayes_probs, nn_probs, rating_probs, y):
        """
        通过最小化Brier Score学习非负融合权重
        约束: weights >= 0, sum(weights) = 1
        """
        from scipy.optimize import minimize

        def brier(weights):
            w = np.clip(weights, 0, 1)
            w = w / (w.sum() + 1e-10)
            pred = (w[0] * bayes_probs + w[1] * nn_probs + w[2] * rating_probs)
            return np.mean((pred - y) ** 2)

        result = minimize(
            brier,
            x0=[1/3, 1/3, 1/3],
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
        )

        self.weights = np.clip(result.x, 0, 1)
        self.weights = self.weights / (self.weights.sum() + 1e-10)
        self.fitted = True

        print(f"\n  [Fusion] Learned weights (non-negative, sum=1):")
        print(f"    Bayesian:        {self.weights[0]:.4f}")
        print(f"    Neural Embed:    {self.weights[1]:.4f}")
        print(f"    Rating Systems:  {self.weights[2]:.4f}")

        return self

    def predict(self, bayes_prob, nn_prob, rating_prob):
        """加权平均融合"""
        if not self.fitted:
            raise ValueError("Fusion model not fitted yet")

        scalar_input = np.isscalar(bayes_prob)
        if scalar_input:
            bayes_prob = np.array([bayes_prob])
            nn_prob = np.array([nn_prob])
            rating_prob = np.array([rating_prob])

        pred = (self.weights[0] * bayes_prob +
                self.weights[1] * nn_prob +
                self.weights[2] * rating_prob)

        return float(pred[0]) if scalar_input else pred


# ==============================================================
# 第5部分: 交叉验证
# ==============================================================

class TimeSeriesCV:
    """
    时间序列交叉验证
    Fold 1: train 1985-2000, val 2001-2005
    Fold 2: train 1985-2005, val 2006-2010
    Fold 3: train 1985-2010, val 2011-2015
    Fold 4: train 1985-2015, val 2016-2020
    Fold 5: train 1985-2020, val 2021-2025
    """

    def __init__(self, fold_years=None):
        if fold_years is None:
            self.folds = [
                (1985, 2000, 2001, 2005),
                (1985, 2005, 2006, 2010),
                (1985, 2010, 2011, 2015),
                (1985, 2015, 2016, 2020),
                (1985, 2020, 2021, 2025),
            ]
        else:
            self.folds = fold_years

    def split(self, games_df):
        """生成训练/验证索引"""
        for train_start, train_end, val_start, val_end in self.folds:
            train_mask = (games_df.Season >= train_start) & (games_df.Season <= train_end)
            val_mask = (games_df.Season >= val_start) & (games_df.Season <= val_end)
            yield train_mask.values, val_mask.values


def cv_predict(data_loader, bayes_model, nn_model, rating_system,
               fusion_model=None, verbose=True):
    """
    时间序列CV获取OOF预测
    返回各视角的OOF预测 + 真实标签
    """
    games = data_loader.all_games
    team_id_to_idx = data_loader.team_id_to_idx
    n_teams = data_loader.n_teams

    # 准备特征
    team_a_idxs = np.array([team_id_to_idx[tid] for tid in games.WTeamID])
    team_b_idxs = np.array([team_id_to_idx[tid] for tid in games.LTeamID])
    y = np.ones(len(games))  # WTeamID always wins in the data

    home_indicator = np.where(
        games.WLoc == 'H', 1,
        np.where(games.WLoc == 'A', -1, 0)
    )

    seasons = games.Season.values
    day_nums = games.DayNum.values

    # OOF预测
    bayes_oof = np.full(len(games), np.nan)
    nn_oof = np.full(len(games), np.nan)
    rating_oof = np.full(len(games), np.nan)

    tscv = TimeSeriesCV()
    fold = 0

    for train_mask, val_mask in tscv.split(games):
        fold += 1
        n_train = train_mask.sum()
        n_val = val_mask.sum()
        if verbose:
            print(f"\n  [CV] Fold {fold}: train={n_train}, val={n_val}")

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # ---- 视角1: 贝叶斯 ----
        if verbose:
            print("  [CV] Training Bayesian model...")
        t0 = time.time()
        bayes = BayesianStrengthModel(time_decay_lambda=0.005)
        bayes.fit(
            team_a_idxs[train_idx], team_b_idxs[train_idx],
            y[train_idx], home_indicator[train_idx],
            seasons=seasons[train_idx], day_nums=day_nums[train_idx],
            current_season=seasons[val_idx].max() if len(val_idx) > 0 else 2026,
            n_teams_full=n_teams
        )
        bayes_oof[val_idx] = bayes.predict_batch(
            team_a_idxs[val_idx], team_b_idxs[val_idx], home_indicator[val_idx]
        )
        if verbose:
            print(f"         Done in {time.time() - t0:.1f}s")

        # ---- 视角2: 神经嵌入 ----
        if verbose:
            print("  [CV] Training Neural Embedding model...")
        t0 = time.time()
        nn = NeuralEmbeddingModel(n_teams, embed_dim=32)
        nn.fit(
            team_a_idxs[train_idx], team_b_idxs[train_idx],
            y[train_idx],
            n_epochs=30, batch_size=256, verbose=verbose
        )
        nn_oof[val_idx] = nn.predict_batch(
            team_a_idxs[val_idx], team_b_idxs[val_idx]
        )
        if verbose:
            print(f"         Done in {time.time() - t0:.1f}s")

        # ---- 视角3: 评分系统 ----
        if verbose:
            print("  [CV] Computing Rating Systems...")
        t0 = time.time()

        train_games = games.iloc[train_idx]
        ratings = RatingSystemEnsemble()
        ratings._setup(n_teams, team_id_to_idx)
        ratings.compute_all_ratings(
            train_games[train_games.is_tourney == 0],
            train_games[train_games.is_tourney == 1],
            regular_season_detail=data_loader.regular_season_detail,
            tourney_detail=data_loader.tourney_detail,
            current_season=seasons[val_idx].max() if len(val_idx) > 0 else 2026
        )
        ratings.fit_weights(team_a_idxs[train_idx], team_b_idxs[train_idx], y[train_idx])

        rating_oof[val_idx] = ratings.predict_batch(
            team_a_idxs[val_idx], team_b_idxs[val_idx]
        )
        if verbose:
            print(f"         Done in {time.time() - t0:.1f}s")

        # 各视角CV分数
        bayes_brier = brier_score_loss(y[val_idx], bayes_oof[val_idx])
        nn_brier = brier_score_loss(y[val_idx], nn_oof[val_idx])
        rating_brier = brier_score_loss(y[val_idx], rating_oof[val_idx])
        if verbose:
            print(f"  [CV] Fold {fold} Brier: Bayes={bayes_brier:.4f}, "
                  f"NN={nn_brier:.4f}, Rating={rating_brier:.4f}")

    # 完整OOF分数
    valid = ~np.isnan(bayes_oof)
    bayes_brier = brier_score_loss(y[valid], bayes_oof[valid])
    nn_brier = brier_score_loss(y[valid], nn_oof[valid])
    rating_brier = brier_score_loss(y[valid], rating_oof[valid])

    print(f"\n{'=' * 50}")
    print(f"OOF Brier Scores:")
    print(f"  Bayesian:        {bayes_brier:.4f}")
    print(f"  Neural Embedding:{nn_brier:.4f}")
    print(f"  Rating Systems:  {rating_brier:.4f}")

    # 融合
    if fusion_model is not None:
        fusion_idx = valid
        fusion_model.fit(
            bayes_oof[fusion_idx],
            nn_oof[fusion_idx],
            rating_oof[fusion_idx],
            y[fusion_idx]
        )
        fusion_pred = fusion_model.predict(
            bayes_oof[fusion_idx],
            nn_oof[fusion_idx],
            rating_oof[fusion_idx]
        )
        fusion_brier = brier_score_loss(y[fusion_idx], fusion_pred)
        fusion_coef = fusion_model.weights if fusion_model.fitted else None
        print(f"  Fusion:          {fusion_brier:.4f}")
        print(f"{'=' * 50}")
    else:
        fusion_coef = None

    return {
        'bayes_oof': bayes_oof,
        'nn_oof': nn_oof,
        'rating_oof': rating_oof,
        'y_true': y,
        'bayes_brier': bayes_brier,
        'nn_brier': nn_brier,
        'rating_brier': rating_brier,
        'fusion_coef': fusion_coef,
        'fusion_brier': fusion_brier if fusion_model is not None else None,
    }


# ==============================================================
# 第6部分: 2026 锦标赛预测
# ==============================================================

def parse_submission_pairs(data_dir=DATA_DIR):
    """
    解析SampleSubmissionStage2.csv，分离男/女对阵
    男子TeamID范围: 1101-1481
    女子TeamID范围: 3101-3481
    """
    sub_path = os.path.join(data_dir, 'SampleSubmissionStage2.csv')
    sample_sub = pd.read_csv(sub_path)

    # 分辨男/女
    men_mask = sample_sub['ID'].str.split('_').str[1].astype(int) < 2000
    men_sub = sample_sub[men_mask].copy()
    women_sub = sample_sub[~men_mask].copy()

    def parse_pairs(df):
        pairs = []
        for row_id in df['ID']:
            parts = row_id.split('_')
            season, team_a, team_b = int(parts[0]), int(parts[1]), int(parts[2])
            pairs.append((team_a, team_b))
        return pairs

    return {
        'M': (parse_pairs(men_sub), men_sub),
        'W': (parse_pairs(women_sub), women_sub),
    }


def predict_2026(data_loader, bayes, nn_model, rating_system, fusion_model,
                 pairs, verbose=True):
    """
    使用全量训练的模型预测2026年所有对阵
    每个视角先用全量数据重新训练
    """
    games = data_loader.all_games
    team_id_to_idx = data_loader.team_id_to_idx
    n_teams = data_loader.n_teams

    team_a_idxs = np.array([team_id_to_idx[tid] for tid in games.WTeamID])
    team_b_idxs = np.array([team_id_to_idx[tid] for tid in games.LTeamID])
    y = np.ones(len(games))
    home_indicator = np.where(games.WLoc == 'H', 1,
                              np.where(games.WLoc == 'A', -1, 0))
    seasons = games.Season.values
    day_nums = games.DayNum.values

    if verbose:
        print("\n" + "=" * 50)
        print("全量数据重训练(1985-2025)...")
        print("=" * 50)

    # ---- 贝叶斯全量训练 ----
    if verbose:
        print("\n[Retrain] Bayesian strength model...")
    t0 = time.time()
    bayes.fit(
        team_a_idxs, team_b_idxs, y, home_indicator,
        seasons=seasons, day_nums=day_nums, current_season=2026,
        n_teams_full=n_teams
    )
    if verbose:
        print(f"         Done in {time.time() - t0:.1f}s")
        top10 = np.argsort(-bayes.theta)[:10]
        print("\n  Top 10 Teams (Bayesian θ):")
        for rank, idx in enumerate(top10, 1):
            tid = data_loader.team_idx_to_id[idx]
            name = data_loader.team_id_to_name.get(tid, f"Team_{tid}")
            print(f"    {rank}. {name:25s} θ={bayes.theta[idx]:.3f}")

    # ---- 神经嵌入全量训练 ----
    if verbose:
        print("\n[Retrain] Neural embedding model...")
    t0 = time.time()
    nn_model.fit(
        team_a_idxs, team_b_idxs, y,
        n_epochs=50, batch_size=256, verbose=verbose
    )
    if verbose:
        print(f"         Done in {time.time() - t0:.1f}s")

    # ---- 评分系统全量训练 ----
    if verbose:
        print("\n[Retrain] Rating systems...")
    t0 = time.time()
    rating_system._setup(n_teams, team_id_to_idx)
    rating_system.compute_all_ratings(
        games[games.is_tourney == 0],
        games[games.is_tourney == 1],
        regular_season_detail=data_loader.regular_season_detail,
        tourney_detail=data_loader.tourney_detail,
        current_season=2026
    )
    rating_system.fit_weights(team_a_idxs, team_b_idxs, y)
    if verbose:
        print(f"         Done in {time.time() - t0:.1f}s")

    # ---- 2026预测 ----
    if verbose:
        print(f"\n[Predict] 2026 tournament ({len(pairs)} matchups)...")

    predictions = []
    for team_a, team_b in pairs:
        a_idx = team_id_to_idx.get(team_a, -1)
        b_idx = team_id_to_idx.get(team_b, -1)

        if a_idx < 0 or b_idx < 0:
            predictions.append(0.5)
            continue

        p_b = bayes.predict(a_idx, b_idx, home=0)
        p_n = nn_model.predict(a_idx, b_idx)
        p_r = rating_system.predict(a_idx, b_idx)
        p_f = fusion_model.predict(p_b, p_n, p_r)

        predictions.append(p_f)

    return np.array(predictions)


# ==============================================================
# 主流程
# ==============================================================

def main():
    print("=" * 60)
    print("方案B: 多视角融合 — 贝叶斯 + 神经嵌入 + 评分系统")
    print("NCAA March Madness 2026 概率预测")
    print("=" * 60)

    # 先解析所有对阵(男/女分离)
    print("\n[准备] 解析提交文件，分离男/女对阵...")
    all_pairs = parse_submission_pairs()
    for g in ['M', 'W']:
        pairs, df = all_pairs[g]
        print(f"  {g}: {len(pairs)} 场对阵")

    all_results = {}

    for gender in ['M', 'W']:
        print(f"\n{'#' * 60}")
        print(f"# 处理 {gender} 组 ({'Men' if gender == 'M' else 'Women'})")
        print(f"{'#' * 60}")

        # 1) 数据加载
        print(f"\n[1/6] 数据加载...")
        t0 = time.time()
        dl = DataLoader(gender)
        print(f"       {len(dl.all_games)} 场比赛, {dl.n_teams} 支球队")
        print(f"       OK ({time.time() - t0:.1f}s)")

        # 2) 初始化模型
        print(f"\n[2/6] 初始化模型...")
        bayes = BayesianStrengthModel(time_decay_lambda=0.005)
        nn_model = NeuralEmbeddingModel(dl.n_teams, embed_dim=32)
        rating_system = RatingSystemEnsemble()
        fusion = FusionModel()

        # 3) CV + OOF预测
        print(f"\n[3/6] 时间序列交叉验证(1985-2025, 5-fold)...")
        t0 = time.time()
        cv_results = cv_predict(dl, bayes, nn_model, rating_system,
                                fusion_model=fusion, verbose=True)
        cv_results['time'] = time.time() - t0

        # 4) 获取该性别的2026对阵
        pairs, sample_sub = all_pairs[gender]
        print(f"\n[4/6] 2026年锦标赛对阵: {len(pairs)} 场")

        # 5) 全量训练 + 2026预测
        print(f"\n[5/6] 全量训练 + 2026预测...")
        t0 = time.time()
        preds_2026 = predict_2026(dl, bayes, nn_model, rating_system,
                                  fusion, pairs, verbose=True)
        predict_time = time.time() - t0
        print(f"       Done ({predict_time:.1f}s)")

        # 6) 生成该性别提交部分
        print(f"\n[6/6] 保存{gender}组预测结果...")
        sample_sub['Pred'] = preds_2026
        sub_path = os.path.join(OUTPUT_DIR, f'submission_{gender}.csv')
        sample_sub.to_csv(sub_path, index=False)
        print(f"       已保存: {sub_path}")
        print(f"       预测分布: min={preds_2026.min():.3f}, "
              f"max={preds_2026.max():.3f}, mean={preds_2026.mean():.3f}, "
              f"std={preds_2026.std():.3f}")

        all_results[gender] = {
            'data': dl,
            'cv': cv_results,
            'predictions': preds_2026,
        }

    # 合并提交
    print(f"\n{'=' * 60}")
    print("合并男/女提交...")

    men_sub = pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_M.csv'))
    women_sub = pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_W.csv'))
    combined = pd.concat([men_sub, women_sub], ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, 'submission_b.csv'), index=False)

    print(f"  Men rows:   {len(men_sub)}")
    print(f"  Women rows: {len(women_sub)}")
    print(f"  Combined:   {len(combined)}")
    print(f"\n  最终提交: {os.path.join(OUTPUT_DIR, 'submission_b.csv')}")

    # 实验日志
    log_path = os.path.join(OUTPUT_DIR, 'solution_b_log.md')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"# 方案B实验日志\n\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## CV Brier分数\n\n")
        for gender in ['M', 'W']:
            cv = all_results[gender]['cv']
            f.write(f"### {gender}\n")
            f.write(f"| 视角 | Brier |\n")
            f.write(f"|------|-------|\n")
            f.write(f"| 贝叶斯 | {cv['bayes_brier']:.4f} |\n")
            f.write(f"| 神经嵌入 | {cv['nn_brier']:.4f} |\n")
            f.write(f"| 评分系统 | {cv['rating_brier']:.4f} |\n")
            if cv.get('fusion_brier'):
                f.write(f"| 融合 | {cv['fusion_brier']:.4f} |\n")
            f.write(f"\n")

        f.write("## 融合权重\n\n")
        for gender in ['M', 'W']:
            coef = all_results[gender]['cv'].get('fusion_coef')
            if coef is not None:
                f.write(f"### {gender}\n")
                f.write("```\n")
                f.write(f"Bayesian:        {coef[0]:.4f}\n")
                f.write(f"Neural Embed:    {coef[1]:.4f}\n")
                f.write(f"Rating Systems:  {coef[2]:.4f}\n")
                f.write("(non-negative weights, sum = 1)\n")
                f.write("```\n\n")

        f.write("## 运行时间\n\n")
        total_time = sum(v['cv'].get('time', 0) for v in all_results.values())
        f.write(f"总计: {total_time:.0f}s ({total_time / 60:.1f}min)\n\n")

        f.write("## 2026预测统计\n\n")
        for gender in ['M', 'W']:
            preds = all_results[gender]['predictions']
            f.write(f"### {gender}\n")
            f.write(f"| Stat | Value |\n")
            f.write(f"|------|-------|\n")
            f.write(f"| min | {preds.min():.4f} |\n")
            f.write(f"| max | {preds.max():.4f} |\n")
            f.write(f"| mean | {preds.mean():.4f} |\n")
            f.write(f"| std | {preds.std():.4f} |\n\n")

    print(f"\n实验日志: {log_path}")
    print(f"\n{'=' * 60}")
    print("方案B执行完成！")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
