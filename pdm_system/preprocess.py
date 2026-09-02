"""
ai4i2020数据集预处理
面向设备预测性维护的智能分析与预警系统
VSCode PowerShell终端运行，增加路径调试，解决FileNotFoundError
"""
import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

# ===================== 路径配置 =====================
# 打印程序当前运行工作目录，排查路径问题
current_dir = os.getcwd()
print(f"====程序当前工作目录：{current_dir}====")

# 相对路径，csv放在项目根pdm_system文件夹
RAW_CSV_PATH = "ai4i2020.csv"

OUTPUT_DIR = "./data"
LOG_DIR = "./log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ai4i2020_processed.csv")
LOG_FILE = os.path.join(LOG_DIR, "preprocess.log")

# 判断原始csv文件是否存在
if not os.path.exists(RAW_CSV_PATH):
    raise FileNotFoundError(f"未找到原始csv文件！路径：{RAW_CSV_PATH}\n请检查ai4i2020.csv是否放在项目根目录！")

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== 读取原始数据 =====================
logger.info("===== 开始读取原始数据集 ai4i2020.csv =====")
df_raw = pd.read_csv(RAW_CSV_PATH)
logger.info(f"原始数据形状: {df_raw.shape}")
logger.info("缺失值统计：")
logger.info(df_raw.isnull().sum().to_string())

df = df_raw.copy()

# ===================== 特征工程 =====================
logger.info("===== 执行特征工程，构造衍生特征 =====")
df["T_diff[K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power_load"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
max_tool_wear = df["Tool wear [min]"].max()
df["Tool_wear_norm"] = df["Tool wear [min]"] / max_tool_wear

# ===================== IQR四分位异常值截断 =====================
logger.info("===== IQR四分位法进行异常值截断 =====")
numeric_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "T_diff[K]",
    "Power_load",
    "Tool_wear_norm"
]

def clip_by_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)

for col in numeric_cols:
    df[col] = clip_by_iqr(df[col])

# ===================== Type独热编码 =====================
logger.info("===== 对设备类型Type进行One‑Hot独热编码 =====")
df = pd.get_dummies(df, columns=["Type"], prefix="Type")

# ===================== Z‑score标准化 =====================
logger.info("===== Z‑score标准化数值特征 =====")
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ===================== 保存结果 =====================
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
logger.info("===== 预处理全部完成 =====")
logger.info(f"处理后数据形状: {df.shape}")
logger.info(f"预处理后CSV保存路径: {OUTPUT_CSV}")
logger.info(f"运行日志保存路径: {LOG_FILE}")
logger.info("前5行预览:\n" + df.head().to_string())
