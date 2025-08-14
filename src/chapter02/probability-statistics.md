# 概率统计

> "概率论是关于不确定性的数学，统计学是从数据中提取知识的科学。" 
> —— 统计学家布拉德利·埃弗伦

概率统计是处理随机现象和不确定性的数学工具。在数学建模中，它帮助我们量化不确定性、分析随机过程、从数据中推断规律，是现代数据科学和机器学习的理论基础。

## 概率论基础

### 概率空间

#### 基本概念

**样本空间（Sample Space）**：所有可能结果的集合，记作 \\(\Omega\\)。

**事件（Event）**：样本空间的子集，通常用大写字母 \\(A, B, C\\) 表示。

**事件域（σ-代数）**：满足一定条件的事件集合 \\(\mathcal{F}\\)：
1. \\(\Omega \in \mathcal{F}\\)
2. 若 \\(A \in \mathcal{F}\\)，则 \\(A^c \in \mathcal{F}\\)
3. 若 \\(A_1, A_2, \ldots \in \mathcal{F}\\)，则 \\(\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}\\)

**概率测度**：函数 \\(P: \mathcal{F} \rightarrow [0,1]\\)，满足概率公理。

#### 概率公理

**公理1（非负性）**：对任意事件 \\(A\\)，\\(P(A) \geq 0\\)

**公理2（归一性）**：\\(P(\Omega) = 1\\)

**公理3（可列可加性）**：对于两两互不相交的事件序列 \\(\{A_i\}\\)：
\\[P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)\\]

#### 概率的基本性质

1. **空集概率**：\\(P(\emptyset) = 0\\)
2. **补集概率**：\\(P(A^c) = 1 - P(A)\\)
3. **单调性**：若 \\(A \subseteq B\\)，则 \\(P(A) \leq P(B)\\)
4. **加法公式**：\\(P(A \cup B) = P(A) + P(B) - P(A \cap B)\\)
5. **包含排斥原理**：
   \\[P(A_1 \cup A_2 \cup \cdots \cup A_n) = \sum_{i} P(A_i) - \sum_{i<j} P(A_i \cap A_j) + \cdots + (-1)^{n+1} P(A_1 \cap \cdots \cap A_n)\\]

### 条件概率与独立性

#### 条件概率

事件 \\(A\\) 在事件 \\(B\\) 发生条件下的概率：
\\[P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0\\]

#### 乘法公式

\\[P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)\\]

**一般形式**：
\\[P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1)P(A_2|A_1)P(A_3|A_1 \cap A_2) \cdots P(A_n|A_1 \cap \cdots \cap A_{n-1})\\]

#### 全概率公式

设 \\(\{B_i\}\\) 是样本空间的一个分割，则对任意事件 \\(A\\)：
\\[P(A) = \sum_{i} P(A|B_i)P(B_i)\\]

#### 贝叶斯定理

\\[P(B_j|A) = \frac{P(A|B_j)P(B_j)}{\sum_{i} P(A|B_i)P(B_i)}\\]

**意义**：
- \\(P(B_j)\\)：先验概率
- \\(P(B_j|A)\\)：后验概率
- \\(P(A|B_j)\\)：似然函数

#### 独立性

**两事件独立**：\\(P(A \cap B) = P(A)P(B)\\)

**等价条件**：
- \\(P(A|B) = P(A)\\)（当 \\(P(B) > 0\\)）
- \\(P(B|A) = P(B)\\)（当 \\(P(A) > 0\\)）

**多事件独立**：
- **两两独立**：任意两个事件独立
- **相互独立**：任意子集的交事件概率等于各事件概率的乘积

### 古典概型与几何概型

#### 古典概型

**条件**：
1. 有限个等可能的基本事件
2. 每个基本事件发生的概率相等

**概率计算**：
\\[P(A) = \frac{\text{事件A包含的基本事件数}}{\text{基本事件总数}} = \frac{|A|}{|\Omega|}\\]

#### 排列组合

**排列数**：从 \\(n\\) 个不同元素中取 \\(r\\) 个元素的排列数
\\[A_n^r = P_n^r = \frac{n!}{(n-r)!}\\]

**组合数**：从 \\(n\\) 个不同元素中取 \\(r\\) 个元素的组合数
\\[C_n^r = \binom{n}{r} = \frac{n!}{r!(n-r)!}\\]

**重要公式**：
- \\(\binom{n}{r} = \binom{n}{n-r}\\)
- \\(\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}\\)
- \\((x+y)^n = \sum_{k=0}^n \binom{n}{k} x^k y^{n-k}\\)

#### 几何概型

当样本空间是连续的几何区域时：
\\[P(A) = \frac{\text{区域A的测度}}{\text{样本空间的测度}}\\]

**测度**可以是长度、面积、体积等。

## 随机变量

### 随机变量的概念

**定义**：随机变量是定义在概率空间上的实值函数：
\\[X: \Omega \rightarrow \mathbb{R}\\]

**分布函数**：
\\[F(x) = P(X \leq x), \quad x \in \mathbb{R}\\]

**性质**：
1. **单调性**：\\(F(x)\\) 单调不减
2. **右连续性**：\\(F(x+0) = F(x)\\)
3. **极限性**：\\(\lim_{x \to -\infty} F(x) = 0\\)，\\(\lim_{x \to +\infty} F(x) = 1\\)

### 离散型随机变量

#### 概率质量函数

\\[p(x_i) = P(X = x_i), \quad i = 1, 2, \ldots\\]

**性质**：
- \\(p(x_i) \geq 0\\)
- \\(\sum_i p(x_i) = 1\\)

#### 常见离散分布

**1. 伯努利分布 \\(B(1, p)\\)**
\\[P(X = k) = \begin{cases} 
p & k = 1 \\
1-p & k = 0
\end{cases}\\]

**2. 二项分布 \\(B(n, p)\\)**
\\[P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n\\]

**3. 几何分布 \\(\text{Geo}(p)\\)**
\\[P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, \ldots\\]

**4. 泊松分布 \\(\text{Poisson}(\lambda)\\)**
\\[P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots\\]

**泊松近似**：当 \\(n\\) 很大，\\(p\\) 很小，\\(np = \lambda\\) 适中时：
\\[B(n, p) \approx \text{Poisson}(\lambda)\\]

### 连续型随机变量

#### 概率密度函数

如果存在非负函数 \\(f(x)\\) 使得：
\\[F(x) = \int_{-\infty}^x f(t) dt\\]

则称 \\(f(x)\\) 为概率密度函数。

**性质**：
- \\(f(x) \geq 0\\)
- \\(\int_{-\infty}^{+\infty} f(x) dx = 1\\)
- \\(P(a < X \leq b) = \int_a^b f(x) dx\\)

#### 常见连续分布

**1. 均匀分布 \\(U(a, b)\\)**
\\[f(x) = \begin{cases}
\frac{1}{b-a} & a \leq x \leq b \\
0 & \text{其他}
\end{cases}\\]

**2. 指数分布 \\(\text{Exp}(\lambda)\\)**
\\[f(x) = \begin{cases}
\lambda e^{-\lambda x} & x \geq 0 \\
0 & x < 0
\end{cases}\\]

**无记忆性**：\\(P(X > s+t | X > s) = P(X > t)\\)

**3. 正态分布 \\(N(\mu, \sigma^2)\\)**
\\[f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}\\]

**标准正态分布** \\(N(0, 1)\\)：
\\[\varphi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}\\]

**标准化**：若 \\(X \sim N(\mu, \sigma^2)\\)，则 \\(Z = \frac{X-\mu}{\sigma} \sim N(0, 1)\\)

**4. 伽马分布 \\(\text{Gamma}(\alpha, \beta)\\)**
\\[f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0\\]

其中 \\(\Gamma(\alpha) = \int_0^{\infty} t^{\alpha-1} e^{-t} dt\\) 是伽马函数。

**5. 卡方分布 \\(\chi^2(n)\\)**
\\[f(x) = \frac{1}{2^{n/2}\Gamma(n/2)} x^{n/2-1} e^{-x/2}, \quad x > 0\\]

**6. t分布 \\(t(n)\\)**
\\[f(x) = \frac{\Gamma((n+1)/2)}{\sqrt{n\pi}\Gamma(n/2)} \left(1 + \frac{x^2}{n}\right)^{-(n+1)/2}\\]

**7. F分布 \\(F(m, n)\\)**
\\[f(x) = \frac{\Gamma((m+n)/2)}{\Gamma(m/2)\Gamma(n/2)} \left(\frac{m}{n}\right)^{m/2} \frac{x^{m/2-1}}{(1 + \frac{m}{n}x)^{(m+n)/2}}, \quad x > 0\\]

### 随机变量的数字特征

#### 数学期望

**离散型**：
\\[E[X] = \sum_i x_i P(X = x_i)\\]

**连续型**：
\\[E[X] = \int_{-\infty}^{+\infty} x f(x) dx\\]

**性质**：
1. **线性性**：\\(E[aX + bY] = aE[X] + bE[Y]\\)
2. **常数**：\\(E[c] = c\\)
3. **独立性**：若 \\(X, Y\\) 独立，则 \\(E[XY] = E[X]E[Y]\\)

#### 方差

\\[\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2\\]

**性质**：
1. \\(\text{Var}(aX + b) = a^2 \text{Var}(X)\\)
2. 若 \\(X, Y\\) 独立，则 \\(\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)\\)

**标准差**：\\(\sigma(X) = \sqrt{\text{Var}(X)}\\)

#### 高阶矩

**k阶原点矩**：\\(\mu_k = E[X^k]\\)

**k阶中心矩**：\\(\nu_k = E[(X - E[X])^k]\\)

**偏度（Skewness）**：
\\[\text{Skew}(X) = \frac{E[(X - \mu)^3]}{\sigma^3}\\]

**峰度（Kurtosis）**：
\\[\text{Kurt}(X) = \frac{E[(X - \mu)^4]}{\sigma^4}\\]

#### 协方差和相关系数

**协方差**：
\\[\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]\\]

**相关系数**：
\\[\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}\\]

**性质**：
- \\(-1 \leq \rho(X, Y) \leq 1\\)
- \\(|\rho(X, Y)| = 1\\) 当且仅当 \\(X, Y\\) 线性相关
- \\(\rho(X, Y) = 0\\) 称为不相关

### 多维随机变量

#### 联合分布

**离散型**：
\\[p(x_i, y_j) = P(X = x_i, Y = y_j)\\]

**连续型**：
\\[F(x, y) = P(X \leq x, Y \leq y) = \int_{-\infty}^x \int_{-\infty}^y f(u, v) dudv\\]

#### 边际分布

**离散型**：
\\[p_X(x_i) = \sum_j p(x_i, y_j)\\]

**连续型**：
\\[f_X(x) = \int_{-\infty}^{+\infty} f(x, y) dy\\]

#### 条件分布

**离散型**：
\\[P(X = x_i | Y = y_j) = \frac{p(x_i, y_j)}{p_Y(y_j)}\\]

**连续型**：
\\[f_{X|Y}(x|y) = \frac{f(x, y)}{f_Y(y)}\\]

#### 独立性

随机变量 \\(X, Y\\) 独立当且仅当：
\\[f(x, y) = f_X(x) f_Y(y)\\]

对所有 \\(x, y\\) 成立。

## 大数定律与中心极限定理

### 收敛性概念

#### 依概率收敛

\\[X_n \xrightarrow{P} X \iff \lim_{n \to \infty} P(|X_n - X| > \epsilon) = 0, \quad \forall \epsilon > 0\\]

#### 几乎必然收敛

\\[X_n \xrightarrow{a.s.} X \iff P(\lim_{n \to \infty} X_n = X) = 1\\]

#### 依分布收敛

\\[X_n \xrightarrow{d} X \iff \lim_{n \to \infty} F_n(x) = F(x)\\]

在 \\(F(x)\\) 的连续点处成立。

### 大数定律

#### 弱大数定律（辛钦大数定律）

设 \\(\{X_n\}\\) 独立同分布，且 \\(E[X_1] = \mu\\) 存在，则：
\\[\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{P} \mu\\]

#### 强大数定律（柯尔莫哥洛夫强大数定律）

设 \\(\{X_n\}\\) 独立同分布，且 \\(E[X_1] = \mu\\) 存在，则：
\\[\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{a.s.} \mu\\]

#### 贝努利大数定律

设 \\(S_n\\) 是 \\(n\\) 次独立重复试验中事件 \\(A\\) 发生的次数，\\(P(A) = p\\)，则：
\\[\frac{S_n}{n} \xrightarrow{P} p\\]

### 中心极限定理

#### 独立同分布中心极限定理（Lindeberg-Lévy定理）

设 \\(\{X_n\}\\) 独立同分布，\\(E[X_1] = \mu\\)，\\(\text{Var}(X_1) = \sigma^2 < \infty\\)，则：
\\[\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} N(0, 1)\\]

#### 棣莫弗-拉普拉斯定理

设 \\(S_n \sim B(n, p)\\)，则当 \\(n \to \infty\\) 时：
\\[\frac{S_n - np}{\sqrt{np(1-p)}} \xrightarrow{d} N(0, 1)\\]

#### 李雅普诺夫中心极限定理

对于独立但不同分布的随机变量序列，在满足李雅普诺夫条件下，标准化的和仍趋向于标准正态分布。

### 应用举例

#### 质量控制

在生产过程中，产品的某项指标 \\(X \sim N(\mu, \sigma^2)\\)。通过样本均值 \\(\bar{X}\\) 来监控过程：

**控制图**：
- 中心线：\\(\mu\\)
- 控制限：\\(\mu \pm 3\frac{\sigma}{\sqrt{n}}\\)

**原理**：由中心极限定理，\\(\bar{X} \sim N(\mu, \frac{\sigma^2}{n})\\)

#### 民意调查

估计支持率 \\(p\\)，样本量为 \\(n\\)，样本支持率为 \\(\hat{p}\\)：

**置信区间**（近似）：
\\[\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\\]

## 参数估计

### 点估计

#### 矩估计法

**原理**：用样本矩估计总体矩

**k阶样本矩**：
\\[A_k = \frac{1}{n} \sum_{i=1}^n X_i^k\\]

**步骤**：
1. 建立总体矩与参数的关系
2. 用样本矩代替总体矩
3. 解方程得到参数估计

**例子**：正态分布 \\(N(\mu, \sigma^2)\\)
- \\(E[X] = \mu \Rightarrow \hat{\mu} = \bar{X}\\)
- \\(\text{Var}(X) = \sigma^2 \Rightarrow \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2\\)

#### 最大似然估计法

**似然函数**：
\\[L(\theta) = \prod_{i=1}^n f(x_i; \theta)\\]

**对数似然函数**：
\\[\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)\\]

**最大似然估计**：
\\[\hat{\theta} = \arg\max_\theta L(\theta) = \arg\max_\theta \ell(\theta)\\]

**求解方法**：
\\[\frac{d\ell(\theta)}{d\theta} = 0\\]

**例子**：指数分布 \\(\text{Exp}(\lambda)\\)
\\[f(x; \lambda) = \lambda e^{-\lambda x}, \quad x > 0\\]
\\[\ell(\lambda) = n\ln\lambda - \lambda\sum_{i=1}^n x_i\\]
\\[\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0\\]
\\[\hat{\lambda} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}\\]

#### 贝叶斯估计

**贝叶斯公式**：
\\[\pi(\theta|x) = \frac{f(x|\theta)\pi(\theta)}{m(x)}\\]

其中：
- \\(\pi(\theta)\\)：先验分布
- \\(\pi(\theta|x)\\)：后验分布
- \\(f(x|\theta)\\)：似然函数
- \\(m(x) = \int f(x|\theta)\pi(\theta)d\theta\\)：边际分布

**点估计**：
- **后验均值**：\\(\hat{\theta}_B = E[\theta|x]\\)
- **后验中位数**：使 \\(P(\theta \leq \hat{\theta}_B|x) = 0.5\\)
- **后验众数**：使 \\(\pi(\theta|x)\\) 最大

### 估计量的评价标准

#### 无偏性

\\[E[\hat{\theta}] = \theta\\]

**例子**：
- \\(\bar{X}\\) 是 \\(\mu\\) 的无偏估计
- \\(S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2\\) 是 \\(\sigma^2\\) 的无偏估计

#### 有效性

在所有无偏估计中，方差最小的估计称为有效估计。

**Cramér-Rao不等式**：
\\[\text{Var}(\hat{\theta}) \geq \frac{1}{nI(\theta)}\\]

其中 \\(I(\theta) = E\left[-\frac{\partial^2 \ln f(X;\theta)}{\partial \theta^2}\right]\\) 是Fisher信息量。

#### 一致性

\\[\hat{\theta}_n \xrightarrow{P} \theta \quad \text{或} \quad \hat{\theta}_n \xrightarrow{a.s.} \theta\\]

### 区间估计

#### 置信区间

对于参数 \\(\theta\\)，如果随机区间 \\([\hat{\theta}_L, \hat{\theta}_U]\\) 满足：
\\[P(\hat{\theta}_L \leq \theta \leq \hat{\theta}_U) = 1 - \alpha\\]

则称其为 \\(\theta\\) 的置信度为 \\(1-\alpha\\) 的置信区间。

#### 正态总体的区间估计

**均值 \\(\mu\\) 的置信区间**（\\(\sigma\\) 已知）：
\\[\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\\]

**均值 \\(\mu\\) 的置信区间**（\\(\sigma\\) 未知）：
\\[\bar{X} \pm t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}}\\]

**方差 \\(\sigma^2\\) 的置信区间**：
\\[\left[\frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)}, \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}\right]\\]

#### 大样本置信区间

当样本量较大时，基于中心极限定理：
\\[\hat{\theta} \pm z_{\alpha/2} \sqrt{\text{Var}(\hat{\theta})}\\]

## 假设检验

### 基本概念

#### 假设的陈述

**原假设**：\\(H_0: \theta = \theta_0\\)

**备择假设**：
- 双侧：\\(H_1: \theta \neq \theta_0\\)
- 单侧：\\(H_1: \theta > \theta_0\\) 或 \\(H_1: \theta < \theta_0\\)

#### 两类错误

**第一类错误（α错误）**：拒绝真的 \\(H_0\\)
\\[\alpha = P(\text{拒绝}H_0 | H_0\text{为真})\\]

**第二类错误（β错误）**：接受假的 \\(H_0\\)
\\[\beta = P(\text{接受}H_0 | H_1\text{为真})\\]

**功效（Power）**：
\\[1 - \beta = P(\text{拒绝}H_0 | H_1\text{为真})\\]

#### 检验统计量与拒绝域

**检验统计量**：\\(T = T(X_1, X_2, \ldots, X_n)\\)

**拒绝域**：使得拒绝 \\(H_0\\) 的 \\(T\\) 值的集合

**临界值**：拒绝域的边界

### 单个正态总体的检验

#### 均值的检验

**1. Z检验**（\\(\sigma\\) 已知）
\\[H_0: \mu = \mu_0 \quad \text{vs} \quad H_1: \mu \neq \mu_0\\]

检验统计量：
\\[Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \sim N(0, 1)\\]

拒绝域：\\(|Z| > z_{\alpha/2}\\)

**2. t检验**（\\(\sigma\\) 未知）
\\[H_0: \mu = \mu_0 \quad \text{vs} \quad H_1: \mu \neq \mu_0\\]

检验统计量：
\\[t = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t(n-1)\\]

拒绝域：\\(|t| > t_{\alpha/2}(n-1)\\)

#### 方差的检验

\\[H_0: \sigma^2 = \sigma_0^2 \quad \text{vs} \quad H_1: \sigma^2 \neq \sigma_0^2\\]

检验统计量：
\\[\chi^2 = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2(n-1)\\]

拒绝域：\\(\chi^2 < \chi^2_{1-\alpha/2}(n-1)\\) 或 \\(\chi^2 > \chi^2_{\alpha/2}(n-1)\\)

### 两个正态总体的检验

#### 均值差的检验

**等方差情况**：
\\[H_0: \mu_1 = \mu_2 \quad \text{vs} \quad H_1: \mu_1 \neq \mu_2\\]

检验统计量：
\\[t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t(n_1 + n_2 - 2)\\]

其中 \\(S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1 + n_2 - 2}\\)

**不等方差情况（Welch检验）**：
\\[t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}}\\]

自由度：
\\[\nu = \frac{(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2})^2}{\frac{S_1^4}{n_1^2(n_1-1)} + \frac{S_2^4}{n_2^2(n_2-1)}}\\]

#### 方差比的检验

\\[H_0: \sigma_1^2 = \sigma_2^2 \quad \text{vs} \quad H_1: \sigma_1^2 \neq \sigma_2^2\\]

检验统计量：
\\[F = \frac{S_1^2}{S_2^2} \sim F(n_1-1, n_2-1)\\]

拒绝域：\\(F < F_{1-\alpha/2}(n_1-1, n_2-1)\\) 或 \\(F > F_{\alpha/2}(n_1-1, n_2-1)\\)

### 非参数检验

#### 符号检验

用于检验中位数：
\\[H_0: M = M_0 \quad \text{vs} \quad H_1: M \neq M_0\\]

检验统计量：正号的个数 \\(S^+ \sim B(n, 0.5)\\)

#### Wilcoxon符号秩检验

考虑差值的大小信息：
1. 计算 \\(|X_i - M_0|\\) 并排秩
2. 赋予符号得到符号秩
3. 计算正符号秩和 \\(W^+\\)

#### Mann-Whitney U检验

用于两样本位置参数的比较：
\\[U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1\\]

其中 \\(R_1\\) 是第一组样本的秩和。

### p值方法

**p值**：在 \\(H_0\\) 成立条件下，观察到当前检验统计量值或更极端值的概率。

**决策规则**：
- 若 \\(p < \alpha\\)，拒绝 \\(H_0\\)
- 若 \\(p \geq \alpha\\)，不拒绝 \\(H_0\\)

**优点**：
- 提供了证据强度的度量
- 不依赖于预先设定的显著性水平

## 方差分析

### 单因素方差分析

#### 模型

\\[X_{ij} = \mu + \alpha_i + \epsilon_{ij}\\]

其中：
- \\(i = 1, 2, \ldots, k\\)（处理组数）
- \\(j = 1, 2, \ldots, n_i\\)（第i组样本量）
- \\(\alpha_i\\) 是第i个处理效应
- \\(\epsilon_{ij} \sim N(0, \sigma^2)\\) 独立

#### 假设

\\[H_0: \alpha_1 = \alpha_2 = \cdots = \alpha_k = 0\\]
\\[H_1: \text{至少有一个} \alpha_i \neq 0\\]

#### 平方和分解

**总平方和**：
\\[SS_T = \sum_{i=1}^k \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{..})^2\\]

**组间平方和**：
\\[SS_A = \sum_{i=1}^k n_i (\bar{X}_{i.} - \bar{X}_{..})^2\\]

**组内平方和**：
\\[SS_E = \sum_{i=1}^k \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{i.})^2\\]

**关系**：\\(SS_T = SS_A + SS_E\\)

#### F检验

\\[F = \frac{MS_A}{MS_E} = \frac{SS_A/(k-1)}{SS_E/(N-k)} \sim F(k-1, N-k)\\]

其中 \\(N = \sum_{i=1}^k n_i\\)

拒绝域：\\(F > F_\alpha(k-1, N-k)\\)

### 双因素方差分析

#### 无交互作用模型

\\[X_{ij} = \mu + \alpha_i + \beta_j + \epsilon_{ij}\\]

#### 有交互作用模型

\\[X_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}\\]

其中 \\((\alpha\beta)_{ij}\\) 是交互作用效应。

## 回归分析

### 一元线性回归

#### 模型

\\[Y_i = \beta_0 + \beta_1 x_i + \epsilon_i, \quad i = 1, 2, \ldots, n\\]

其中 \\(\epsilon_i \sim N(0, \sigma^2)\\) 独立。

#### 最小二乘估计

最小化残差平方和：
\\[Q(\beta_0, \beta_1) = \sum_{i=1}^n (Y_i - \beta_0 - \beta_1 x_i)^2\\]

解得：
\\[\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}\\]

\\[\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{x}\\]

#### 回归方程的显著性检验

**假设**：
\\[H_0: \beta_1 = 0 \quad \text{vs} \quad H_1: \beta_1 \neq 0\\]

**F检验**：
\\[F = \frac{SS_R/1}{SS_E/(n-2)} = \frac{MS_R}{MS_E} \sim F(1, n-2)\\]

**决定系数**：
\\[R^2 = \frac{SS_R}{SS_T} = 1 - \frac{SS_E}{SS_T}\\]

其中：
- \\(SS_T = \sum_{i=1}^n (Y_i - \bar{Y})^2\\)（总平方和）
- \\(SS_R = \sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2\\)（回归平方和）
- \\(SS_E = \sum_{i=1}^n (Y_i - \hat{Y}_i)^2\\)（残差平方和）

#### 参数的置信区间

\\[\hat{\beta}_1 \pm t_{\alpha/2}(n-2) \sqrt{\frac{MS_E}{S_{xx}}}\\]

\\[\hat{\beta}_0 \pm t_{\alpha/2}(n-2) \sqrt{MS_E \left(\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}\right)}\\]

#### 预测

**点预测**：\\(\hat{Y}_0 = \hat{\beta}_0 + \hat{\beta}_1 x_0\\)

**均值的置信区间**：
\\[\hat{Y}_0 \pm t_{\alpha/2}(n-2) \sqrt{MS_E \left(\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}\right)}\\]

**个体值的预测区间**：
\\[\hat{Y}_0 \pm t_{\alpha/2}(n-2) \sqrt{MS_E \left(1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}\right)}\\]

### 多元线性回归

#### 模型

\\[\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}\\]

其中：
\\[\mathbf{Y} = \begin{pmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{pmatrix}, \quad \mathbf{X} = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{pmatrix}, \quad \boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{pmatrix}\\]

#### 最小二乘估计

\\[\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}\\]

**性质**：
- \\(E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}\\)（无偏性）
- \\(\text{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\\)

#### 回归诊断

**残差分析**：
- 残差：\\(e_i = Y_i - \hat{Y}_i\\)
- 标准化残差：\\(r_i = \frac{e_i}{\sqrt{MS_E}}\\)
- 学生化残差：\\(t_i = \frac{e_i}{\sqrt{MS_E(1-h_{ii})}}\\)

**异常值检测**：
- 杠杆值：\\(h_{ii} = (\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T)_{ii}\\)
- Cook距离：\\(D_i = \frac{r_i^2}{p+1} \cdot \frac{h_{ii}}{1-h_{ii}}\\)

## 概率统计在建模中的应用

### 蒙特卡罗方法

#### 基本思想

利用随机抽样解决数学问题，特别是求解复杂的积分和优化问题。

#### 简单蒙特卡罗积分

估计积分 \\(I = \int_a^b g(x) dx\\)：

1. 在 \\([a,b]\\) 上均匀抽样：\\(x_i \sim U(a, b)\\)
2. 计算：\\(\hat{I} = (b-a) \frac{1}{n} \sum_{i=1}^n g(x_i)\\)

**理论基础**：\\(E[\hat{I}] = I\\)，\\(\text{Var}(\hat{I}) = \frac{(b-a)^2}{n} \text{Var}(g(X))\\)

#### 重要性抽样

当被积函数在某些区域值较大时，使用重要性抽样提高效率：

\\[\int g(x) f(x) dx = \int \frac{g(x) f(x)}{h(x)} h(x) dx = E_{h}\left[\frac{g(X) f(X)}{h(X)}\right]\\]

选择合适的重要性函数 \\(h(x)\\) 可以减小方差。

#### 马尔可夫链蒙特卡罗（MCMC）

**Metropolis-Hastings算法**：

1. 给定当前状态 \\(x^{(t)}\\)
2. 从提议分布 \\(q(x^*|x^{(t)})\\) 中产生候选状态 \\(x^*\\)
3. 计算接受概率：\\(\alpha = \min\left(1, \frac{\pi(x^*) q(x^{(t)}|x^*)}{\pi(x^{(t)}) q(x^*|x^{(t)})}\right)\\)
4. 以概率 \\(\alpha\\) 接受 \\(x^*\\)，否则保持 \\(x^{(t)}\\)

**Gibbs抽样**：对于多维分布，逐个从条件分布中抽样。

### 排队论

#### M/M/1排队系统

**假设**：
- 到达过程：泊松过程，强度 \\(\lambda\\)
- 服务时间：指数分布，参数 \\(\mu\\)
- 单个服务台
- 无限容量，先到先服务

**稳态概率**：
\\[\pi_n = (1-\rho)\rho^n, \quad n = 0, 1, 2, \ldots\\]

其中 \\(\rho = \frac{\lambda}{\mu} < 1\\)

**性能指标**：
- 平均队长：\\(L = \frac{\rho}{1-\rho}\\)
- 平均等待时间：\\(W = \frac{\rho}{\mu(1-\rho)}\\)
- Little公式：\\(L = \lambda W\\)

#### M/M/c排队系统

**稳态概率**：
\\[\pi_n = \begin{cases}
\frac{\rho^n}{n!} \pi_0 & n = 0, 1, \ldots, c \\
\frac{\rho^n}{c! c^{n-c}} \pi_0 & n > c
\end{cases}\\]

其中 \\(\pi_0^{-1} = \sum_{n=0}^c \frac{\rho^n}{n!} + \frac{\rho^c}{c!} \cdot \frac{c}{c-\rho}\\)

### 可靠性理论

#### 可靠性函数

\\[R(t) = P(T > t) = 1 - F(t)\\]

其中 \\(T\\) 是产品寿命。

#### 失效率函数

\\[\lambda(t) = \frac{f(t)}{R(t)} = \frac{f(t)}{1-F(t)}\\]

#### 常用寿命分布

**指数分布**：
- 失效率：\\(\lambda(t) = \lambda\\)（常数）
- 无记忆性：\\(P(T > s+t | T > s) = P(T > t)\\)

**威布尔分布**：
\\[f(t) = \frac{\beta}{\eta} \left(\frac{t}{\eta}\right)^{\beta-1} e^{-(t/\eta)^\beta}\\]

- 失效率：\\(\lambda(t) = \frac{\beta}{\eta} \left(\frac{t}{\eta}\right)^{\beta-1}\\)
- \\(\beta < 1\\)：递减失效率（早期失效）
- \\(\beta = 1\\)：常数失效率（随机失效）
- \\(\beta > 1\\)：递增失效率（磨损失效）

#### 系统可靠性

**串联系统**：
\\[R_s(t) = \prod_{i=1}^n R_i(t)\\]

**并联系统**：
\\[R_s(t) = 1 - \prod_{i=1}^n [1 - R_i(t)]\\]

**k-out-of-n系统**：
\\[R_s(t) = \sum_{i=k}^n \binom{n}{i} [R(t)]^i [1-R(t)]^{n-i}\\]

### 金融数学

#### 期权定价模型

**Black-Scholes模型**：

假设股价遵循几何布朗运动：
\\[dS_t = \mu S_t dt + \sigma S_t dW_t\\]

**期权定价公式**：

欧式看涨期权价格：
\\[C = S_0 N(d_1) - K e^{-rT} N(d_2)\\]

其中：
\\[d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}\\]
\\[d_2 = d_1 - \sigma\sqrt{T}\\]

#### 风险度量

**VaR（Value at Risk）**：

在给定置信水平下的最大可能损失：
\\[P(\text{损失} \leq \text{VaR}_\alpha) = \alpha\\]

**CVaR（Conditional VaR）**：

超过VaR的条件期望损失：
\\[\text{CVaR}_\alpha = E[\text{损失} | \text{损失} > \text{VaR}_\alpha]\\]

### 生物统计

#### 生存分析

**生存函数**：
\\[S(t) = P(T > t)\\]

**风险函数**：
\\[h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t | T \geq t)}{\Delta t}\\]

**Kaplan-Meier估计**：

对于有截尾数据的生存函数估计：
\\[\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)\\]

其中 \\(d_i\\) 是在时刻 \\(t_i\\) 的死亡数，\\(n_i\\) 是风险集大小。

**Cox比例风险模型**：
\\[h(t|x) = h_0(t) \exp(\beta^T x)\\]

其中 \\(h_0(t)\\) 是基准风险函数。

## 小结

概率统计为数学建模提供了处理不确定性的强大工具：

1. **概率论**：建立了随机现象的数学框架
2. **统计推断**：从样本数据推断总体特征
3. **假设检验**：科学决策的统计方法
4. **回归分析**：建立变量间的定量关系
5. **随机过程**：描述动态随机系统

掌握概率统计的关键在于：
- 理解概率的公理化定义和基本性质
- 熟练掌握常用分布及其应用场景
- 掌握统计推断的基本方法和原理
- 能够选择合适的统计方法解决实际问题
- 理解统计结果的含义和局限性

概率统计与其他数学工具结合，为现代数据科学、机器学习和人工智能提供了坚实的理论基础。
