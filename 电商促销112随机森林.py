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
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

#数据生成
np.random.seed(42)
n = 500
data = {
    "促销类型": np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
    "折扣率": np.round(np.random.uniform(0, 0.3, n), 2),
    "广告费用": np.random.randint(1000, 5001, n),
}
data["销售额"] = (
    10000 
    + 7000 * data["折扣率"] 
    + 0.8 * data["广告费用"] 
    + 500 * (data["促销类型"] == 0)
    + np.random.normal(0, 200, n)
)
df = pd.DataFrame(data)

#数据探索
print("\n描述性统计：")
print(df.describe())

#多元线性回归
# 准备数据（添加常数项）
X_ols = sm.add_constant(df[["折扣率", "广告费用", "促销类型"]])
y = df["销售额"]

# 拟合模型
ols_model = sm.OLS(y, X_ols).fit()
print("\n多元线性回归结果：")
print(ols_model.summary())

#随机森林模型
# 准备数据（不需要添加常数项）
X = df[["折扣率", "广告费用", "促销类型"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 评估模型
y_pred_rf = rf_model.predict(X_test)
print(f"\n随机森林测试集R²: {r2_score(y_test, y_pred_rf):.3f}")
print(f"随机森林测试集MSE: {mean_squared_error(y_test, y_pred_rf):.1f}")

# 特征重要性
importance = pd.DataFrame({
    "特征": X.columns,
    "重要性": rf_model.feature_importances_
}).sort_values("重要性", ascending=False)
print("\n特征重要性排序：")
print(importance)

#模型对比
# 关键修复：正确准备线性回归的测试集数据（必须包含常数项）
X_test_ols = sm.add_constant(X_test)
y_pred_ols = ols_model.predict(X_test_ols)

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_ols, alpha=0.5, label="线性回归")
plt.scatter(y_test, y_pred_rf, alpha=0.5, label="随机森林")
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2)
plt.xlabel("实际销售额")
plt.ylabel("预测销售额")
plt.title("模型预测效果对比")
plt.legend()
plt.grid(True)
plt.savefig("model_comparison.png")
plt.show()

#可视化分析
# 特征重要性（修复警告）
plt.figure(figsize=(10, 6))
sns.barplot(
    x="重要性", 
    y="特征", 
    hue="特征",
    data=importance,
    palette="Blues_d",
    legend=False
)
plt.title("随机森林特征重要性分析")
plt.savefig("feature_importance.png")
plt.show()

# 部分依赖图
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    rf_model, X, features=[0, 1, (0, 1)],
    feature_names=X.columns, ax=ax
)
plt.suptitle("部分依赖图分析")
plt.tight_layout()
plt.savefig("partial_dependence.png")
plt.show()

#保存结果
# 保存数据到桌面
df.to_csv(r"C:\Users\w\Desktop\promotion_data.csv", index=False)
print("\n数据已保存到桌面：promotion_data.csv")
