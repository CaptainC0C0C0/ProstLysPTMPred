# 赖氨酸多标签修饰位点预测器 (ProstLysPTM Predictor)

一个基于 **ProstT5 + MLPSTAAP + 理化性质** 特征融合，使用 **LightGBM Classifier Chain + Label Powerset + MLP** 集成学习的多标签预测工具，支持**乙酰化(A)、巴豆酰化(C)、甲基化(M)、琥珀酰化(S)** 四种赖氨酸修饰的联合预测。

---

## 项目特色

- **高维特征融合**：ProstT5 (1024维) + MLPSTAAP (46维) + 理化性质 (147维) = **1217维**
- **不平衡处理**：NCR + Cluster Centroids + MDNDO 三阶段采样平衡
- **集成学习**：60条 Classifier Chain + Label Powerset + MLP
- **优化策略**：差分进化参数优化 + MR3 针对性后处理
- **可解释性**：完整 SHAP 分析（全局重要性 + 位置特异性）
- **用户友好**：带图形界面的独立预测器

---

## 项目最终结构

```bash
ProstLysPTMPred/
├── Fused_ProstT5_MLPSTAAP_Physchem/          # 特征融合结果
├── Balanced_MDNDO_NCR_CC/                    # 最终平衡数据集
├── Final_Models/                             # 训练好的模型（必须保留）
├── shap_results/                             # SHAP 可解释性分析结果
├── Under/                                    # NCR & ClusterCentroids 模块
├── Train Dataset/                            # 训练集原始数据
├── Test Dataset/                             # 测试集原始数据
├── evaluation_indicators/                    # 评估指标模块
├── Feature_Extraction_and_Fusion.py
├── MDNDO_NCRCC.py
├── LightGBM_based_Classifier_and_SHAP_Analysis.py
├── Predictor.py                              # 图形化预测器（可单独运行）
├── requirements.txt
└── README.md
```

## 快速安装

### 推荐安装方式

```bash
# 1. 克隆仓库
git clone https://github.com/CaptainC0C0C0/ProstLysPTMPred.git
cd ProstLysPTMPred

# 2. 创建并激活虚拟环境
python -m venv venv

# Windows 系统：
venv\Scripts\activate
# Linux / macOS 系统：
# source venv/bin/activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装依赖
pip install -r requirements.txt
```

## 使用流程

方式一：直接使用预测器（推荐大多数用户）
```bash
python Predictor.py
```
方式二：从头完整训练（可选）
```bash
# 1. 特征融合
python Feature_Extraction_and_Fusion.py

# 2. 数据采样平衡
python MDNDO_NCRCC.py

# 3. 模型训练 + SHAP 分析
python LightGBM_based_Classifier_and_SHAP_Analysis.py
