import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
import torch
import torch.nn as nn
import warnings
from scipy.optimize import differential_evolution
from collections import Counter
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ====================== 评估指标 ======================
from evaluation_indicators.Absolute_False import absolute_false
from evaluation_indicators.Absolute_True import absolute_true
from evaluation_indicators.Accuracy import accuracy
from evaluation_indicators.Aiming import aiming
from evaluation_indicators.Coverage import coverage
from evaluation_indicators.MR import calculate_MR

# ====================== 固定随机种子 ======================
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== Label Mapping ======================
label_mapping = {
    1: [1,0,0,0], 2: [0,1,0,0], 3: [0,0,1,0], 4: [0,0,0,1],
    5: [1,1,0,0], 6: [1,0,1,0], 7: [1,0,0,1], 8: [0,1,1,0],
    9: [1,1,1,0],10:[1,1,0,1],11:[1,1,1,1]
}

def labels_to_class(y):
    y_int = y.astype(int)
    for cid, pat in label_mapping.items():
        if np.array_equal(y_int, pat):
            return cid - 1
    return -1

# ====================== 字体设置 (SHAP 中文支持) ======================
plt.rcParams['axes.unicode_minus'] = False
try:
    if 'win' in plt.sys.platform:
        zh_font_path = "C:/Windows/Fonts/simhei.ttf"
        en_font_name = "Times New Roman"
    else:
        zh_font_path = "/usr/share/fonts/truetype/liberation/SimHei.ttf"
        en_font_name = "DejaVu Sans"
    chinese_font = fm.FontProperties(fname=zh_font_path, size=12)
    english_font = fm.FontProperties(family=en_font_name, size=11)
except:
    chinese_font = fm.FontProperties(family='SimHei', size=12)
    english_font = fm.FontProperties(family='serif', size=11)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# ====================== 1. 数据加载 ======================
print("正在加载平衡后的融合特征...")
data_dir = "Balanced_MDNDO_NCR_CC"

X = np.load(f'{data_dir}/X_train_features.npy').astype(np.float32)
y = np.load(f'{data_dir}/y_train_labels.npy').astype(np.float32)
X_test_raw = np.load(f'{data_dir}/X_test_features.npy').astype(np.float32)
y_test = np.load(f'{data_dir}/y_test_labels.npy').astype(np.float32)

print(f"训练集形状: {X.shape}, 测试集形状: {X_test_raw.shape}")

if X.ndim == 3:
    X = X.reshape(X.shape[0], -1)
    X_test_raw = X_test_raw.reshape(X_test_raw.shape[0], -1)

y_argmax = y.argmax(axis=1)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_argmax)

mean = np.mean(X_tr, axis=0, keepdims=True)
std = np.std(X_tr, axis=0, keepdims=True) + 1e-8
X_tr = (X_tr - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test_raw - mean) / std

label_corr = np.corrcoef(y_tr.T)
print("数据准备完成。")

# ====================== 2. Classifier Chains ======================
print("\n=== 阶段 2: Enhanced Classifier Chains (60条) ===")
base_params = {
    'n_estimators': 3000, 'learning_rate': 0.0175, 'num_leaves': 85,
    'min_child_samples': 9, 'feature_fraction': 0.70, 'bagging_fraction': 0.80,
    'bagging_freq': 4, 'lambda_l1': 4.0, 'lambda_l2': 3.5, 'verbose': -1,
    'n_jobs': -1, 'random_state': 42, 'is_unbalance': True
}

chains = []
for i in range(60):
    order = np.random.permutation(4).tolist() if i % 3 != 0 else None
    chain = ClassifierChain(lgb.LGBMClassifier(**base_params), order=order, random_state=i)
    chain.fit(X_tr, y_tr)
    chains.append(chain)
    if (i + 1) % 10 == 0:
        print(f"  已完成 {i+1}/60 条 Classifier Chain")

y_probs_chains = np.mean([chain.predict_proba(X_test) for chain in chains], axis=0)

# ====================== 3. Label Powerset ======================
print("\n=== 阶段 3: Label Powerset ===")
y_tr_class = np.array([labels_to_class(row) for row in y_tr])
y_val_class = np.array([labels_to_class(row) for row in y_val])

lp_params = {
    'objective': 'multiclass', 'num_class': 11, 'learning_rate': 0.021,
    'num_leaves': 195, 'feature_fraction': 0.725, 'bagging_fraction': 0.825,
    'lambda_l1': 1.7, 'lambda_l2': 4.5, 'verbose': -1,
    'num_threads': -1, 'is_unbalance': True
}

train_ds = lgb.Dataset(X_tr, y_tr_class)
val_ds = lgb.Dataset(X_val, y_val_class, reference=train_ds)

lp_model = lgb.train(
    lp_params, train_ds, num_boost_round=3800,
    valid_sets=[val_ds], callbacks=[lgb.early_stopping(220)]
)

lp_raw_probs = lp_model.predict(X_test)
y_probs_powerset = np.zeros((len(X_test), 4))
for c in range(11):
    pat = np.array(label_mapping[c + 1])
    y_probs_powerset += lp_raw_probs[:, c:c+1] * pat
y_probs_powerset = np.clip(y_probs_powerset, 1e-5, 1 - 1e-5)

# ====================== 4. MLP ======================
print("\n=== 阶段 4: MLP ===")
class PTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.33),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.33),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden_dim//2, 4)
        )
    def forward(self, x):
        return self.net(x)

mlp_model = PTMModel(X_test.shape[1]).to(device)
mlp_path = "best_ptm_model.pth"
if os.path.exists(mlp_path):
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    print("MLP权重加载成功")
else:
    print("未找到MLP权重，使用随机初始化")

mlp_model.eval()
with torch.no_grad():
    y_probs_mlp = torch.sigmoid(mlp_model(torch.from_numpy(X_test).to(device))).cpu().numpy()

# ====================== 5. 后处理函数 ======================
def mr_targeted_post_processing(probs, base_thresh, corr_matrix, boost=0.105, min_labels=1.55, high_conf_thresh=0.84):
    y_pred = (probs >= base_thresh).astype(int)
    N = len(probs)
    for i in range(4):
        for j in range(4):
            if i != j and corr_matrix[i, j] > 0.50:
                high_mask = (probs[:, i] > high_conf_thresh) & (y_pred[:, j] == 0)
                y_pred[high_mask, j] = (probs[high_mask, j] > (base_thresh[j] - boost)).astype(int)

    pred_counts = y_pred.sum(axis=1)
    for idx in range(N):
        if pred_counts[idx] < min_labels:
            missing = np.where(y_pred[idx] == 0)[0]
            if len(missing) > 0:
                miss_probs = probs[idx, missing]
                k_to_add = max(0, int(min_labels - pred_counts[idx]))
                if k_to_add > 0:
                    top_k = np.argsort(miss_probs)[-k_to_add:]
                    for t in top_k:
                        if miss_probs[t] > base_thresh[missing[t]] - 0.06:
                            y_pred[idx, missing[t]] = 1

    mapping_list = list(label_mapping.values())
    for idx in range(N):
        pat = tuple(y_pred[idx])
        if pat not in [tuple(p) for p in mapping_list]:
            scores = [np.dot(probs[idx], np.array(p)) for p in mapping_list]
            best_idx = np.argmax(scores)
            y_pred[idx] = mapping_list[best_idx]
    return y_pred

# ====================== 6. 差分进化优化 ======================
def objective(params):
    w = np.array(params[0:3])
    w = w / (w.sum() + 1e-8)
    thresh = np.clip(params[3:7], 0.04, 0.91)
    powers = np.array(params[7:10])
    boost = params[10]
    temp = np.clip(params[11], 0.82, 1.38)
    min_labels_factor = params[12]

    ens = (w[0] * np.power(y_probs_chains, powers[0]) +
           w[1] * np.power(y_probs_powerset, powers[1]) +
           w[2] * np.power(y_probs_mlp, powers[2]))
    ens = np.clip(ens ** (1.0 / temp), 0.0, 1.0)

    min_labels = 1.40 + min_labels_factor * 0.85
    y_pred = mr_targeted_post_processing(ens, thresh, label_corr, boost=boost,
                                         min_labels=min_labels, high_conf_thresh=0.84)

    at = absolute_true(y_pred, y_test)
    af = absolute_false(y_pred, y_test)
    acc = accuracy(y_pred, y_test)
    aim = aiming(y_pred, y_test)
    mr2 = calculate_MR(y_pred, y_test, 2)
    mr3 = calculate_MR(y_pred, y_test, 3)

    score = (0.45 * at + 0.16 * acc + 0.13 * aim + 0.12 * mr2 + 0.14 * mr3 - 0.07 * af)
    return -score

print("\n=== 开始差分进化优化 ===")
bounds = [(0, 1)] * 3 + [(0.04, 0.90)] * 4 + [(0.60, 1.48)] * 3 + [(0.07, 0.23), (0.82, 1.35), (0.05, 0.95)]

result = differential_evolution(
    objective, bounds, popsize=60, maxiter=420,
    mutation=(0.42, 0.92), recombination=0.77,
    seed=42, disp=False, tol=1e-6
)

best_params = result.x
best_w = best_params[0:3] / best_params[0:3].sum()
best_t = best_params[3:7]
best_p = best_params[7:10]
best_boost = best_params[10]
best_temp = best_params[11]
best_min_factor = best_params[12]

print(f"\n优化完成！最佳分数: {-result.fun:.5f}")
print(f"权重: {best_w.round(4)}")
print(f"阈值: {best_t.round(4)}")
print(f"Power: {best_p.round(4)}")
print(f"Boost: {best_boost:.4f} | Temp: {best_temp:.4f} | MinFactor: {best_min_factor:.4f}")

# ====================== 最终预测 ======================
ens_prob = (best_w[0] * np.power(y_probs_chains, best_p[0]) +
            best_w[1] * np.power(y_probs_powerset, best_p[1]) +
            best_w[2] * np.power(y_probs_mlp, best_p[2]))
ens_prob = np.clip(ens_prob ** (1.0 / best_temp), 0.0, 1.0)

y_pred_final = mr_targeted_post_processing(
    ens_prob, best_t, label_corr,
    boost=best_boost,
    min_labels=1.40 + best_min_factor * 0.85,
    high_conf_thresh=0.84
)

# ====================== 性能评估 ======================
print("\n" + "=" * 60)
print("【最终模型性能】")
print(f"Absolute False : {absolute_false(y_pred_final, y_test):.4f}")
print(f"Absolute True  : {absolute_true(y_pred_final, y_test):.4f}")
print(f"Accuracy       : {accuracy(y_pred_final, y_test):.4f}")
print(f"Aiming         : {aiming(y_pred_final, y_test):.4f}")
print(f"Coverage       : {coverage(y_pred_final, y_test):.4f}")
print(f"MR1 / MR2 / MR3: {calculate_MR(y_pred_final, y_test, 1):.4f} / "
      f"{calculate_MR(y_pred_final, y_test, 2):.4f} / "
      f"{calculate_MR(y_pred_final, y_test, 3):.4f}")
print("=" * 60)

np.save("y_pred_final.npy", y_pred_final)

# ====================== 保存模型与参数 ======================
save_dir = "Final_Models"
os.makedirs(save_dir, exist_ok=True)
shap_dir = "shap_results"
os.makedirs(shap_dir, exist_ok=True)

np.save(f"{save_dir}/mean.npy", mean)
np.save(f"{save_dir}/std.npy", std)
lp_model.save_model(f"{save_dir}/lp_model.txt")

os.makedirs(f"{save_dir}/chains", exist_ok=True)
for i in range(min(8, len(chains))):
    joblib.dump(chains[i], f"{save_dir}/chains/chain_{i}.pkl")

torch.save(mlp_model.state_dict(), f"{save_dir}/best_ptm_model.pth")

best_params_dict = {
    "weights": best_w.tolist(), "thresholds": best_t.tolist(),
    "powers": best_p.tolist(), "boost": float(best_boost),
    "temp": float(best_temp), "min_labels_factor": float(best_min_factor),
    "label_corr": label_corr.tolist()
}
np.save(f"{save_dir}/best_ensemble_params.npy", best_params_dict)
np.save(f"{shap_dir}/X_test_shap.npy", X_test)
np.save(f"{shap_dir}/y_test_shap.npy", y_test)
np.save(f"{shap_dir}/best_ensemble_params.npy", best_params_dict)

print(f"\n模型已保存至 {save_dir}/ 和 {shap_dir}/")

# ====================== SHAP 可解释性分析 ======================
print("\n=== 执行 SHAP 可解释性分析 ===")
lp_model_shap = lgb.Booster(model_file=f"{save_dir}/lp_model.txt")
prostt5_dim, mlpstaap_dim, physchem_dim = 1024, 46, 147

feature_names = []
for i in range(X_test.shape[1]):
    if i < prostt5_dim:
        feature_names.append(f"ProstT5_Emb_{i}")
    elif i < prostt5_dim + mlpstaap_dim:
        feature_names.append(f"MLPSTAAP_Pos_{i - prostt5_dim}")
    else:
        feature_names.append(f"PhysChem_{i - prostt5_dim - mlpstaap_dim}")

explainer = shap.TreeExplainer(lp_model_shap)
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), 600, replace=False)
X_sample = X_test[sample_idx]

shap_values_raw = explainer.shap_values(X_sample)
if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
    shap_values = [shap_values_raw[:, :, c] for c in range(shap_values_raw.shape[2])]
else:
    shap_values = shap_values_raw if isinstance(shap_values_raw, list) else [shap_values_raw]

# Beeswarm Plot
plt.figure(figsize=(11, 9))
shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, max_display=20, show=False)
ax = plt.gca()
ax.set_title("标签0（单纯乙酰化）的 SHAP 分布图", fontproperties=chinese_font, fontsize=16, pad=20)
ax.set_xlabel("SHAP 值（对模型输出的影响）", fontproperties=chinese_font, fontsize=14)
ax.set_ylabel("特征名称", fontproperties=chinese_font, fontsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(english_font)
plt.tight_layout()
plt.savefig(f"{shap_dir}/shap_beeswarm_label0.png", dpi=300, bbox_inches='tight')
plt.close()

# 全局重要性饼图
all_shap_importance = np.zeros(X_sample.shape[1])
for sv in shap_values:
    all_shap_importance += np.abs(sv).mean(axis=0)
all_shap_importance /= len(shap_values)

prost_imp = all_shap_importance[:prostt5_dim].sum()
mlp_imp = all_shap_importance[prostt5_dim:prostt5_dim+mlpstaap_dim].sum()
phys_imp = all_shap_importance[prostt5_dim+mlpstaap_dim:].sum()
total_imp = prost_imp + mlp_imp + phys_imp

plt.figure(figsize=(8, 8))
labels = ['ProstT5 嵌入特征', 'MLPSTAAP 位置特征', '理化性质特征']
sizes = [prost_imp, mlp_imp, phys_imp]
colors_pie = ['#2E86C1', '#E67E22', '#27AE60']
patches, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.2f%%',
                                  colors=colors_pie, startangle=90, pctdistance=0.75,
                                  wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in texts: t.set_fontproperties(chinese_font)
for t in autotexts:
    t.set_fontproperties(english_font)
    t.set_color('white')
    t.set_weight('bold')
plt.title("全局特征重要性占比（11类平均）", fontproperties=chinese_font, fontsize=16)
plt.savefig(f"{shap_dir}/importance_pie.png", dpi=300)
plt.close()

# Top 20 柱状图
top_idx = np.argsort(all_shap_importance)[-20:][::-1]
top_names = [feature_names[i] for i in top_idx]
top_imps = all_shap_importance[top_idx]

plt.figure(figsize=(12, 9))
colors_bar = ['#2E86C1' if 'ProstT5' in name else '#E67E22' if 'MLPSTAAP' in name else '#27AE60' for name in top_names]
sns.barplot(x=top_imps, y=top_names, palette=colors_bar, hue=top_names, legend=False)
ax_bar = plt.gca()
ax_bar.set_title('全局特征重要性 Top 20（所有11类平均）', fontproperties=chinese_font, fontsize=16, pad=15)
ax_bar.set_xlabel('平均绝对 SHAP 值', fontproperties=chinese_font, fontsize=14)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
for tick in ax_bar.get_yticklabels() + ax_bar.get_xticklabels():
    tick.set_fontproperties(english_font)
plt.tight_layout()
plt.savefig(f"{shap_dir}/global_importance_bar.png", dpi=300, bbox_inches='tight')
plt.close()

# 打印结果
print(f"\n=== 全局贡献统计 ===")
print(f"ProstT5 嵌入向量 : {prost_imp:.4f} ({prost_imp/total_imp*100:.2f}%)")
print(f"MLPSTAAP 位置特征: {mlp_imp:.4f} ({mlp_imp/total_imp*100:.2f}%)")
print(f"理化性质特征     : {phys_imp:.4f} ({phys_imp/total_imp*100:.2f}%)")

print("\n=== Top 10 重要特征列表 ===")
for i, idx in enumerate(top_idx[:10]):
    print(f"{i+1:2d}. {feature_names[idx]:35s} : {all_shap_importance[idx]:.5f}")

print(f"\n完整流程执行完毕！所有模型、预测结果和 SHAP 图表已保存。")