# model-code
多元线性回归模型和随机森林模型
# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro, probplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

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

# 新增代码：导入scipy库进行t检验
from scipy import stats

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

# 保存数据到桌面
df.to_csv(r"C:\Users\w\Desktop\promotion_data.csv", index=False)
print("\n数据已保存到桌面：promotion_data.csv")




