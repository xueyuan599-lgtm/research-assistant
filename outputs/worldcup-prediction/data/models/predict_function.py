
def predict_match(home_team: str, away_team: str,
                  df_features: pd.DataFrame, pipeline: dict,
                  feature_names: list) -> dict:
    """
    预测一场比赛的结果概率

    Parameters
    ----------
    home_team : str
        主队名称（必须与特征矩阵中的球队名称一致）
    away_team : str
        客队名称
    df_features : pd.DataFrame
        包含所有比赛的特征矩阵（用于提取该场比赛的已计算特征）
    pipeline : dict
        最终模型管线（含 model/calibrator 和 best_params）
    feature_names : list
        特征名列表（用于确保特征对齐）

    Returns
    -------
    dict: {"away_win": float, "draw": float, "home_win": float}
    """
    # 找到这场比赛
    mask = (df_features["home_team"] == home_team) & \
           (df_features["away_team"] == away_team)

    if mask.sum() == 0:
        raise ValueError(f"未找到 {home_team} vs {away_team} 的比赛记录")

    # 取最后一场匹配的比赛（可能有多次交锋）
    row = df_features[mask].iloc[-1]

    # 构造特征向量（同预处理流程）
    id_cols = ["date", "home_team", "away_team", "tournament", "source",
               "home_score", "away_score", "home_win", "draw", "target"]
    feature_cols = [c for c in df_features.columns if c not in id_cols]
    cat_cols = ["home_confederation", "away_confederation"]

    feat_dict = {}
    for c in feature_cols:
        if c in cat_cols:
            # 处理分类变量 one-hot
            for fn in feature_names:
                if fn.startswith(c + "_"):
                    expected_val = fn.replace(c + "_", "")
                    feat_dict[fn] = 1.0 if str(row[c]) == expected_val else 0.0
        else:
            val = row[c]
            if pd.isna(val):
                val = 0.0
            feat_dict[c] = float(val)

    # 确保所有特征都存在（缺失补 0）
    X_input = np.array([feat_dict.get(fn, 0.0) for fn in feature_names]).reshape(1, -1)

    # 预测
    if pipeline.get("calibrated", False):
        proba = pipeline["calibrator"].predict_proba(X_input)
    else:
        proba = pipeline["model"].predict_proba(X_input)

    return {
        "away_win": round(float(proba[0, 0]), 6),
        "draw": round(float(proba[0, 1]), 6),
        "home_win": round(float(proba[0, 2]), 6),
    }

# 更简洁的封装版本（直接加载模型文件）
def predict_from_file(home_team: str, away_team: str,
                      feature_csv: str, model_pkl: str,
                      feature_names_json: str) -> dict:
    """从文件加载预测"""
    import json, pickle
    df = pd.read_csv(feature_csv, encoding="utf-8-sig")
    with open(model_pkl, "rb") as f:
        pipeline = pickle.load(f)
    with open(feature_names_json, "r") as f:
        fnames = json.load(f)
    return predict_match(home_team, away_team, df, pipeline, fnames)
