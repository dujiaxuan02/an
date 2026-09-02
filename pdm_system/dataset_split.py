"""
数据集划分脚本
面向设备预测性维护的智能分析与预警系统
分层划分训练集、测试集，保存csv，PowerShell终端输出日志
"""
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# ===================== 路径配置 =====================
INPUT_CSV = "./data/ai4i2020_processed.csv"
OUTPUT_DIR = "./data"
LOG_DIR = "./log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(OUTPUT_DIR, "train_dataset.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test_dataset.csv")
SPLIT_LOG = os.path.join(LOG_DIR, "dataset_split.log")

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(SPLIT_LOG, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== 读取预处理后数据集 =====================
logger.info("===== 开始读取预处理后的数据集 =====")
df = pd.read_csv(INPUT_CSV)
logger.info(f"完整数据集shape：{df.shape}")
logger.info(f"设备故障(Machine failure)分布：\n{df['Machine failure'].value_counts().to_string()}")

# ===================== 分层抽样划分 7:3 训练集/测试集 =====================
X = df.copy()
y = df["Machine failure"]

df_train, df_test = train_test_split(
    X,
    test_size=0.3,
    random_state=42,
    stratify=y
)

logger.info("===== 数据集划分完成 =====")
logger.info(f"训练集 shape: {df_train.shape}")
logger.info(f"测试集 shape: {df_test.shape}")
logger.info(f"训练集故障分布：\n{df_train['Machine failure'].value_counts().to_string()}")
logger.info(f"测试集故障分布：\n{df_test['Machine failure'].value_counts().to_string()}")

# ===================== 保存划分后的csv文件 =====================
df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
df_test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

logger.info(f"训练集保存路径：{TRAIN_CSV}")
logger.info(f"测试集保存路径：{TEST_CSV}")
logger.info(f"划分日志保存路径：{SPLIT_LOG}")
print(f"✅训练集保存：{TRAIN_CSV}")
print(f"✅测试集保存：{TEST_CSV}")
print(f"训练集样本数：{df_train.shape[0]}")
print(f"测试集样本数：{df_test.shape[0]}")
