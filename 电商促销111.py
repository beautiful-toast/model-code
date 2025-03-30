# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import shapiro, probplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
# 导入scipy库进行t检验
from scipy import stats

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 设置随机种子（保证每次生成的数据相同，便于复现）
np.random.seed(42)

# 生成n条数据
n = 211
data = {
    # 促销类型：40%为满减（0），60%为折扣（1）
    "促销类型": np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
    
    # 折扣率：0%~30%，保留两位小数
    "折扣率": np.round(np.random.uniform(0, 0.3, n), 2),
    
    # 广告费用：1000~5000元，均匀分布
    "广告费用": np.random.randint(1000, 5001, n),
}

# 计算销售额（基于线性关系 + 随机噪声）
data["销售额"] = (
    10000  # 基础销售额
    + 7000 * data["折扣率"]  # 折扣率影响
    + 0.8 * data["广告费用"]  # 广告费用影响
    + 500 * (data["促销类型"] == 0)  # 促销类型影响
    + np.random.normal(0, 100, n)  # 随机噪声（均值0，标准差100）
)

# 转换为DataFrame（表格形式）
df = pd.DataFrame(data)

# 查看前10行数据
print("模拟数据示例：")
print(df.head(10))

#输出各字段的均值、标准差、最小值、最大值等统计量。
print("\n描述性统计：")
print(df.describe())

# 1. 折扣率与销售额关系散点图
plt.figure(figsize=(10, 6))
plt.scatter(df["折扣率"], df["销售额"], color="blue", alpha=0.6)
plt.title("折扣率与销售额关系")  # 中文标题
plt.xlabel("折扣率")
plt.ylabel("销售额（元）")
plt.grid(True)
plt.savefig("discount_vs_sales.png")
plt.show()

# 按促销类型分组数据
group0 = df[df["促销类型"] == 0]["销售额"]  # 满减活动（0）
group1 = df[df["促销类型"] == 1]["销售额"]  # 折扣活动（1）

# 检查方差齐性（Levene检验）
levene_test = stats.levene(group0, group1)
print(f"Levene检验结果：统计量={levene_test.statistic:.3f}, p值={levene_test.pvalue:.3f}")

# 根据方差齐性选择t检验方法
if levene_test.pvalue > 0.05:
    # 方差齐，使用标准t检验
    t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=True)
else:
    # 方差不齐，使用Welch's t检验
    t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)

print(f"\n独立样本t检验结果：")
print(f"t值 = {t_stat:.3f}")
print(f"p值 = {p_value:.3f}")

# 判断显著性（α=0.05）
if p_value < 0.05:
    print("结论：促销类型对销售额有显著影响（p < 0.05）")
else:
    print("结论：促销类型对销售额无显著影响（p ≥ 0.05）")

# 改进的箱线图代码
plt.figure(figsize=(10, 6))
sns.boxplot(
    x="促销类型",
    y="销售额",
    hue="促销类型",  # 将 x 变量赋值给 hue
    data=df,
    palette="Set2",  # 颜色主题
    width=0.5,       # 箱体宽度
    flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 8}  # 异常值样式
)
plt.title("促销类型与销售额关系（Python箱线图）", fontsize=14)
plt.xlabel("促销类型（0:满减, 1:折扣）", fontsize=12)
plt.ylabel("销售额（元）", fontsize=12)
plt.xticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("promo_type_vs_sales_boxplot_python.png")
plt.show()

# 3. 广告费用与销售额关系散点图（带回归线）
X = df[["广告费用"]]
y = df["销售额"]

# 拟合线性回归模型
model = LinearRegression()
model.fit(X, y)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(df["广告费用"], df["销售额"], color="green", alpha=0.6)
plt.plot(X, model.predict(X), color="red", linewidth=2)  # 绘制回归线
plt.title("广告费用与销售额关系（带回归线）")  # 中文标题
plt.xlabel("广告费用（元）")
plt.ylabel("销售额（元）")
plt.grid(True)
plt.savefig("ad_cost_vs_sales_regression.png")
plt.show()

# 多元线性回归分析
X = df[["折扣率", "广告费用", "促销类型"]]
X = sm.add_constant(X)
y = df["销售额"]
model = sm.OLS(y, X).fit()
print(model.summary())

# 计算残差
residuals = y - model.predict(X)
df["残差"] = residuals

# 残差分析
# 残差直方图
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor="black")
plt.title("残差分布直方图")
plt.xlabel("残差")
plt.ylabel("频数")
plt.show()

# Q-Q 图
plt.figure(figsize=(10, 6))
probplot(residuals, plot=plt)
plt.title("残差 Q-Q 图")
plt.show()

# Shapiro-Wilk 正态性检验
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk 检验结果：统计量={shapiro_test.statistic:.3f}, p值={shapiro_test.pvalue:.3f}")
if shapiro_test.pvalue > 0.05:
    print("残差服从正态分布（p > 0.05）")
else:
    print("残差不服从正态分布（p ≤ 0.05）")

# 计算 VIF 值
vif_data = pd.DataFrame()
vif_data["变量"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF 值：")
print(vif_data)
print(df[["销售额", "残差"]].head())


# 随机森林模型
# 数据划分
# 数据划分
X = df[["折扣率", "广告费用", "促销类型"]]
y = df["销售额"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 模型评估
y_pred = rf_model.predict(X_test)
print(f"\n随机森林测试集R²: {r2_score(y_test, y_pred):.3f}")
print(f"随机森林测试集MSE: {mean_squared_error(y_test, y_pred):.1f}")

# 特征重要性分析
importance = pd.DataFrame({
    "特征": X.columns,
    "重要性": rf_model.feature_importances_
}).sort_values("重要性", ascending=False)
print("\n特征重要性排序：")
print(importance)

# 保存预测结果
df["随机森林预测"] = rf_model.predict(X)

#可视化分析
# 特征重要性柱状图
plt.figure(figsize=(10, 6))
sns.barplot(
    x="重要性",
    y="特征",
    hue="特征",  # 新增hue参数
    data=importance,
    palette="Blues_d"
)
plt.title("随机森林特征重要性分析")
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.show()

# 部分依赖图 (需要sklearn 0.24+)
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    rf_model, X, features=[0, 1, (0, 1)],
    feature_names=X.columns, ax=ax
)
plt.suptitle("部分依赖图分析")
plt.tight_layout()
plt.savefig("partial_dependence.png")
plt.show()

#模型对比
# 预测结果对比图
plt.figure(figsize=(12, 6))
plt.scatter(
    y_test,
    rf_model.predict(X_test[X_test.index]), 
    alpha=0.5,
    label="线性回归"
)
plt.scatter(y_test,
            y_pred,
            alpha=0.5,
            label="随机森林"
)
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2)
plt.xlabel("实际销售额")
plt.ylabel("预测销售额")
plt.title("模型预测效果对比")
plt.legend()
plt.grid(True)
plt.savefig("model_comparison.png")
plt.show()


# 保存数据到桌面
df.to_csv(r"C:\Users\w\Desktop\promotion_data.csv", index=False)
print("\n数据已保存到桌面：promotion_data.csv")



