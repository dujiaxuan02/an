"""
模型训练模块：随机森林+遗传算法超参数寻优
面向设备预测性维护的智能分析与预警系统
输出最优模型保存至 ./model/rf_best_model.pkl
"""
# =========！必须放在所有import最前面！限制MKL线程，解决forrtl error 200崩溃 =========
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from deap import base, creator, tools, algorithms
import logging

# --------------------------路径配置--------------------------
INPUT_TRAIN = "./data/train_dataset.csv"
MODEL_SAVE_PATH = "./model/rf_best_model.pkl"
os.makedirs("./model", exist_ok=True)
os.makedirs("./log", exist_ok=True)

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)


def eval_rf(individual):
    """
    遗传算法适应度评估函数
    individual: [n_estimators, max_depth, min_samples_split]
    """
    n_estimators = int(individual[0])
    max_depth = int(individual[1])
    min_samples_split = int(individual[2])

    n_estimators = np.clip(n_estimators, 50, 300)
    max_depth = np.clip(max_depth, 5, 30)
    min_samples_split = np.clip(min_samples_split, 2, 20)

    # n_jobs=1！关闭随机森林内部多进程，规避MKL崩溃
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
    # cross_val_score同样 n_jobs=1，禁止多进程
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1", n_jobs=1)
    return (scores.mean(), )


def ga_optimize():
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def gen_n_est():
        return np.random.randint(50, 301)
    def gen_max_d():
        return np.random.randint(5, 31)
    def gen_min_split():
        return np.random.randint(2, 21)

    toolbox.register("attr_n_est", gen_n_est)
    toolbox.register("attr_max_d", gen_max_d)
    toolbox.register("attr_min_split", gen_min_split)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_n_est, toolbox.attr_max_d, toolbox.attr_min_split), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_rf)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[50,5,2], up=[300,30,20], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logger.info("=====开始遗传算法超参数优化====")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=5,
                                   stats=stats, halloffame=hof, verbose=True)

    logger.info("====遗传算法迭代全部完成，读取最优个体====")
    best_ind = hof[0]
    logger.info(f"最优个体参数 n_estimators:{best_ind[0]}, max_depth:{best_ind[1]}, min_samples_split:{best_ind[2]}")
    return best_ind


if __name__ == "__main__":
    # 加载训练集
    df_train = pd.read_csv(INPUT_TRAIN)

    drop_cols = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    X_train_raw = df_train.drop(columns=drop_cols)
    y_train_raw = df_train["Machine failure"]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
    logger.info(f"SMOTE后训练集 X:{X_train.shape}, y:{y_train.shape}")

    best_params = ga_optimize()

    logger.info("开始训练最终随机森林模型")
    best_rf = RandomForestClassifier(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        min_samples_split=int(best_params[2]),
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
    best_rf.fit(X_train, y_train)

    joblib.dump(best_rf, MODEL_SAVE_PATH)
    logger.info(f"最优模型已保存至 {MODEL_SAVE_PATH}")
    print("========训练流程全部结束==========")
