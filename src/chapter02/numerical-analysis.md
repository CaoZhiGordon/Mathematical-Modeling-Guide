# 数值分析

> "数值分析是数学与计算机的桥梁，它将连续的数学转化为离散的计算。" 
> —— 数值分析家劳埃德·特雷费森

数值分析是研究用数值方法求解数学问题的学科。在数学建模中，当解析方法难以求解或不存在时，数值方法提供了强大的计算工具，使得复杂的数学模型能够在计算机上实现并求解。

## 数值分析基础

### 误差理论

#### 误差的分类

**模型误差**：现实问题抽象为数学模型时引入的误差。

**观测误差**：由于测量精度有限而产生的误差。

**截断误差**：将无限过程用有限过程近似时产生的误差。

**舍入误差**：计算机表示实数时的精度限制引起的误差。

#### 绝对误差与相对误差

设 \\(x^*\\) 是精确值，\\(x\\) 是近似值：

**绝对误差**：\\(e = x - x^*\\)

**绝对误差限**：\\(|x - x^*| \leq \varepsilon\\)

**相对误差**：\\(e_r = \frac{x - x^*}{x^*}\\)（当 \\(x^* \neq 0\\)）

**相对误差限**：\\(\left|\frac{x - x^*}{x^*}\right| \leq \varepsilon_r\\)

#### 有效数字

如果近似值 \\(x\\) 的绝对误差限是某一位数的半个单位，则称 \\(x\\) 精确到该位，该位到第一位非零数字的位数称为有效数字的位数。

**定理**：设 \\(x = \pm 0.a_1a_2\cdots a_n \times 10^m\\)（\\(a_1 \neq 0\\)），如果其绝对误差限为 \\(\varepsilon = \frac{1}{2} \times 10^{m-n}\\)，则 \\(x\\) 有 \\(n\\) 位有效数字。

#### 误差传播

**函数误差传播**：设 \\(y = f(x)\\)，当 \\(x\\) 的误差为 \\(\Delta x\\) 时：
\\[\Delta y \approx f'(x) \Delta x\\]

**多元函数误差传播**：设 \\(u = f(x_1, x_2, \ldots, x_n)\\)：
\\[\Delta u \approx \sum_{i=1}^n \frac{\partial f}{\partial x_i} \Delta x_i\\]

**四则运算的误差传播**：

1. **加减法**：\\((x_1 \pm x_2)^* = x_1^* \pm x_2^*\\)
   \\[\varepsilon(x_1 \pm x_2) \leq \varepsilon(x_1) + \varepsilon(x_2)\\]

2. **乘法**：\\((x_1 \cdot x_2)^* = x_1^* \cdot x_2^*\\)
   \\[\varepsilon_r(x_1 \cdot x_2) \leq \varepsilon_r(x_1) + \varepsilon_r(x_2)\\]

3. **除法**：\\((x_1 / x_2)^* = x_1^* / x_2^*\\)
   \\[\varepsilon_r(x_1 / x_2) \leq \varepsilon_r(x_1) + \varepsilon_r(x_2)\\]

### 数值稳定性

#### 条件数

对于问题 \\(y = f(x)\\)，条件数定义为：
\\[\text{cond}(f, x) = \left|\frac{f'(x) \cdot x}{f(x)}\right|\\]

**意义**：
- \\(\text{cond} \ll 1\\)：良态问题
- \\(\text{cond} \gg 1\\)：病态问题

#### 算法稳定性

**前向稳定性**：算法产生的是与精确输入"邻近"的输入对应的精确输出。

**后向稳定性**：算法产生的输出是原问题在"邻近"输入下的精确解。

**数值稳定性**：前向稳定或后向稳定的算法称为数值稳定的。

## 插值与逼近

### 多项式插值

#### 拉格朗日插值

给定 \\(n+1\\) 个不同的节点 \\((x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)\\)，拉格朗日插值多项式为：

\\[L_n(x) = \sum_{k=0}^n y_k l_k(x)\\]

其中拉格朗日基函数：
\\[l_k(x) = \prod_{j=0, j \neq k}^n \frac{x - x_j}{x_k - x_j}\\]

**性质**：
- \\(L_n(x_i) = y_i\\)，\\(i = 0, 1, \ldots, n\\)
- 次数不超过 \\(n\\) 的唯一多项式

#### 牛顿插值

**牛顿前向差分公式**：
\\[N_n(x) = f[x_0] + f[x_0, x_1](x - x_0) + \cdots + f[x_0, x_1, \ldots, x_n](x - x_0)(x - x_1)\cdots(x - x_{n-1})\\]

**差商**的递推定义：
\\[f[x_i] = f(x_i)\\]
\\[f[x_i, x_{i+1}, \ldots, x_{i+k}] = \frac{f[x_{i+1}, \ldots, x_{i+k}] - f[x_i, \ldots, x_{i+k-1}]}{x_{i+k} - x_i}\\]

#### 插值误差

**定理**：设 \\(f(x)\\) 在 \\([a, b]\\) 上有 \\(n+1\\) 阶连续导数，\\((x_0, x_1, \ldots, x_n) \subset [a, b]\\)，则对任意 \\(x \in [a, b]\\)，存在 \\(\xi \in (a, b)\\) 使得：

\\[f(x) - L_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} \prod_{i=0}^n (x - x_i)\\]

#### Runge现象

对于等距节点，当插值次数增加时，插值多项式在区间端点附近可能出现剧烈振荡。

**解决方法**：
1. 使用Chebyshev点：\\(x_k = \cos\frac{(2k+1)\pi}{2(n+1)}\\)，\\(k = 0, 1, \ldots, n\\)
2. 分段插值
3. 样条插值

### 样条插值

#### 三次样条插值

在区间 \\([a, b]\\) 上给定节点 \\(a = x_0 < x_1 < \cdots < x_n = b\\) 和函数值 \\(y_i = f(x_i)\\)，三次样条函数 \\(S(x)\\) 满足：

1. 在每个子区间 \\([x_i, x_{i+1}]\\) 上，\\(S(x)\\) 是三次多项式
2. \\(S(x_i) = y_i\\)，\\(i = 0, 1, \ldots, n\\)
3. \\(S(x)\\) 在 \\([a, b]\\) 上二阶连续可导

设在 \\([x_i, x_{i+1}]\\) 上：
\\[S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3\\]

通过连续性条件和边界条件可以确定所有系数。

**边界条件**：
1. **自然边界条件**：\\(S''(a) = S''(b) = 0\\)
2. **夹紧边界条件**：\\(S'(a) = f'(a)\\)，\\(S'(b) = f'(b)\\)

#### 样条插值的优越性

**定理**：在所有满足插值条件和边界条件的二阶连续可导函数中，三次样条函数使得积分 \\(\int_a^b [f''(x)]^2 dx\\) 最小。

### 最佳逼近

#### 最佳一致逼近

在 \\(C[a, b]\\) 上，寻找 \\(n\\) 次多项式 \\(p_n^*(x)\\) 使得：
\\[\|f - p_n^*\|_\infty = \min_{p_n \in P_n} \|f - p_n\|_\infty\\]

**Chebyshev定理**：\\(p_n^*(x)\\) 是 \\(f(x)\\) 的最佳一致逼近多项式当且仅当 \\(f(x) - p_n^*(x)\\) 在 \\([a, b]\\) 上至少有 \\(n+2\\) 个点达到最大绝对值且正负交替。

#### 最佳平方逼近

寻找函数 \\(g(x)\\) 使得：
\\[\|f - g\|_2^2 = \int_a^b [f(x) - g(x)]^2 \rho(x) dx\\]

最小，其中 \\(\rho(x) > 0\\) 是权函数。

**正交多项式系统**：
\\[\int_a^b \phi_i(x) \phi_j(x) \rho(x) dx = \delta_{ij}\\]

**最佳平方逼近**：
\\[g^*(x) = \sum_{k=0}^n c_k \phi_k(x)\\]

其中 \\(c_k = \int_a^b f(x) \phi_k(x) \rho(x) dx\\)

#### 常用正交多项式

**Legendre多项式**（\\([-1, 1]\\)，\\(\rho(x) = 1\\)）：
\\[P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n}[(x^2 - 1)^n]\\]

**Chebyshev多项式**（\\([-1, 1]\\)，\\(\rho(x) = \frac{1}{\sqrt{1-x^2}}\\)）：
\\[T_n(x) = \cos(n \arccos x)\\]

**Hermite多项式**（\\((-\infty, \infty)\\)，\\(\rho(x) = e^{-x^2}\\)）：
\\[H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n}(e^{-x^2})\\]

**Laguerre多项式**（\\([0, \infty)\\)，\\(\rho(x) = e^{-x}\\)）：
\\[L_n(x) = e^x \frac{d^n}{dx^n}(x^n e^{-x})\\]

## 数值积分与数值微分

### 数值积分

#### Newton-Cotes公式

**基本思想**：用插值多项式近似被积函数。

**一般形式**：
\\[\int_a^b f(x) dx \approx \sum_{i=0}^n A_i f(x_i)\\]

其中 \\(A_i\\) 是积分系数。

#### 低阶Newton-Cotes公式

**梯形公式**（\\(n = 1\\)）：
\\[\int_a^b f(x) dx \approx \frac{b-a}{2}[f(a) + f(b)]\\]

**误差**：\\(R[f] = -\frac{(b-a)^3}{12}f''(\xi)\\)，\\(\xi \in (a, b)\\)

**Simpson公式**（\\(n = 2\\)）：
\\[\int_a^b f(x) dx \approx \frac{b-a}{6}\left[f(a) + 4f\left(\frac{a+b}{2}\right) + f(b)\right]\\]

**误差**：\\(R[f] = -\frac{(b-a)^5}{90}f^{(4)}(\xi)\\)，\\(\xi \in (a, b)\\)

#### 复合积分公式

**复合梯形公式**：
\\[T_n = \frac{h}{2}\left[f(a) + 2\sum_{i=1}^{n-1}f(x_i) + f(b)\right]\\]

其中 \\(h = \frac{b-a}{n}\\)，\\(x_i = a + ih\\)

**误差**：\\(R[f] = -\frac{(b-a)h^2}{12}f''(\xi)\\)

**复合Simpson公式**：
\\[S_n = \frac{h}{6}\left[f(a) + 4\sum_{i=0}^{n-1}f(x_{i+1/2}) + 2\sum_{i=1}^{n-1}f(x_i) + f(b)\right]\\]

**误差**：\\(R[f] = -\frac{(b-a)h^4}{180}f^{(4)}(\xi)\\)

#### Gauss积分公式

**基本思想**：选择最优的节点和权重，使积分公式的代数精度最高。

**Gauss-Legendre积分**：
\\[\int_{-1}^1 f(x) dx \approx \sum_{i=1}^n w_i f(x_i)\\]

其中 \\(x_i\\) 是 \\(n\\) 次Legendre多项式的零点，\\(w_i\\) 是对应的权重。

**代数精度**：Gauss \\(n\\) 点公式具有 \\(2n-1\\) 次代数精度。

**常用Gauss点和权重**：

| \\(n\\) | \\(x_i\\) | \\(w_i\\) |
|-----|-------|-------|
| 2 | \\(\pm\frac{1}{\sqrt{3}}\\) | \\(1\\) |
| 3 | \\(0, \pm\sqrt{\frac{3}{5}}\\) | \\(\frac{8}{9}, \frac{5}{9}\\) |

#### 自适应积分

**基本思想**：根据函数的局部性质自动调整步长。

**Simpson自适应积分算法**：
1. 计算 \\(S_1 = S[a, b]\\)（一个区间的Simpson值）
2. 计算 \\(S_2 = S[a, c] + S[c, b]\\)（两个区间的Simpson值），其中 \\(c = \frac{a+b}{2}\\)
3. 如果 \\(|S_2 - S_1| < 15\varepsilon\\)，则接受 \\(S_2\\)
4. 否则递归处理 \\([a, c]\\) 和 \\([c, b]\\)

### 数值微分

#### 差分公式

**前向差分**：
\\[f'(x_0) \approx \frac{f(x_0 + h) - f(x_0)}{h}\\]

**后向差分**：
\\[f'(x_0) \approx \frac{f(x_0) - f(x_0 - h)}{h}\\]

**中心差分**：
\\[f'(x_0) \approx \frac{f(x_0 + h) - f(x_0 - h)}{2h}\\]

#### 高阶导数

**二阶导数**：
\\[f''(x_0) \approx \frac{f(x_0 - h) - 2f(x_0) + f(x_0 + h)}{h^2}\\]

#### Richardson外推

**基本思想**：利用不同步长的结果消除高阶误差项。

**例子**：对于中心差分公式，有：
\\[f'(x_0) = \frac{f(x_0 + h) - f(x_0 - h)}{2h} - \frac{h^2}{6}f'''(x_0) + O(h^4)\\]

使用步长 \\(h\\) 和 \\(h/2\\)：
\\[D(h) = \frac{f(x_0 + h) - f(x_0 - h)}{2h}\\]
\\[D(h/2) = \frac{f(x_0 + h/2) - f(x_0 - h/2)}{h}\\]

**Richardson外推**：
\\[D = \frac{4D(h/2) - D(h)}{3}\\]

这样可以得到 \\(O(h^4)\\) 精度的导数近似。

## 线性方程组的数值解法

### 直接方法

#### Gauss消元法

**基本思想**：通过行变换将系数矩阵化为上三角矩阵。

**算法步骤**：

1. **消元过程**：对于 \\(k = 1, 2, \ldots, n-1\\)
   - 选择主元 \\(a_{kk}^{(k-1)} \neq 0\\)
   - 计算乘数：\\(m_{ik} = \frac{a_{ik}^{(k-1)}}{a_{kk}^{(k-1)}}\\)，\\(i = k+1, \ldots, n\\)
   - 消元：\\(a_{ij}^{(k)} = a_{ij}^{(k-1)} - m_{ik} a_{kj}^{(k-1)}\\)，\\(j = k+1, \ldots, n\\)
   - 更新右端：\\(b_i^{(k)} = b_i^{(k-1)} - m_{ik} b_k^{(k-1)}\\)

2. **回代过程**：
   \\[x_n = \frac{b_n^{(n-1)}}{a_{nn}^{(n-1)}}\\]
   \\[x_i = \frac{b_i^{(i-1)} - \sum_{j=i+1}^n a_{ij}^{(i-1)} x_j}{a_{ii}^{(i-1)}}\\]

**计算量**：\\(\frac{2n^3}{3} + O(n^2)\\) 次浮点运算

#### 选主元策略

**列主元法**：在第 \\(k\\) 步消元时，选择第 \\(k\\) 列中绝对值最大的元素作为主元。

**全主元法**：在第 \\(k\\) 步消元时，在右下角子矩阵中选择绝对值最大的元素作为主元。

#### LU分解

**定理**：如果矩阵 \\(A\\) 的所有顺序主子式都不为零，则存在唯一的单位下三角矩阵 \\(L\\) 和上三角矩阵 \\(U\\) 使得 \\(A = LU\\)。

**Doolittle分解**：\\(L\\) 的对角元为1
\\[\begin{cases}
u_{ij} = a_{ij} - \sum_{k=1}^{i-1} l_{ik} u_{kj}, & i \leq j \\
l_{ij} = \frac{a_{ij} - \sum_{k=1}^{j-1} l_{ik} u_{kj}}{u_{jj}}, & i > j
\end{cases}\\]

**求解步骤**：
1. 分解：\\(A = LU\\)
2. 前代：\\(Ly = b\\)
3. 回代：\\(Ux = y\\)

#### Cholesky分解

对于对称正定矩阵 \\(A\\)，存在唯一的下三角矩阵 \\(L\\) 使得：
\\[A = LL^T\\]

**算法**：
\\[l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}\\]
\\[l_{ij} = \frac{a_{ij} - \sum_{k=1}^{j-1} l_{ik} l_{jk}}{l_{jj}}, \quad i > j\\]

**优点**：
- 计算量约为LU分解的一半
- 数值稳定性好
- 存储量少

### 迭代方法

#### 基本迭代格式

将方程组 \\(Ax = b\\) 改写为 \\(x = Bx + f\\) 的形式，构造迭代格式：
\\[x^{(k+1)} = Bx^{(k)} + f\\]

其中 \\(B\\) 是迭代矩阵，\\(f\\) 是常向量。

#### Jacobi迭代

\\[x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}\right)\\]

**矩阵形式**：\\(x^{(k+1)} = -D^{-1}(L + U)x^{(k)} + D^{-1}b\\)

其中 \\(A = D + L + U\\)（\\(D\\)：对角，\\(L\\)：下三角，\\(U\\)：上三角）

#### Gauss-Seidel迭代

\\[x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^n a_{ij} x_j^{(k)}\right)\\]

**矩阵形式**：\\(x^{(k+1)} = -(D + L)^{-1}Ux^{(k)} + (D + L)^{-1}b\\)

#### SOR方法

\\[x_i^{(k+1)} = (1-\omega)x_i^{(k)} + \frac{\omega}{a_{ii}}\left(b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^n a_{ij} x_j^{(k)}\right)\\]

其中 \\(\omega\\) 是松弛因子。

**最优松弛因子**：对于某些特殊矩阵，最优松弛因子为：
\\[\omega_{\text{opt}} = \frac{2}{1 + \sqrt{1 - \rho^2}}\\]

其中 \\(\rho\\) 是Jacobi迭代矩阵的谱半径。

#### 收敛性分析

**定理**：迭代格式 \\(x^{(k+1)} = Bx^{(k)} + f\\) 收敛当且仅当迭代矩阵 \\(B\\) 的谱半径 \\(\rho(B) < 1\\)。

**收敛速度**：收敛速度由 \\(\rho(B)\\) 决定，\\(\rho(B)\\) 越小收敛越快。

**充分条件**：
- **严格对角占优**：\\(|a_{ii}| > \sum_{j \neq i} |a_{ij}|\\)
- **对称正定**：对于Gauss-Seidel迭代

### 病态问题与条件数

#### 条件数

对于可逆矩阵 \\(A\\)，条件数定义为：
\\[\text{cond}(A) = \|A\| \|A^{-1}\|\\]

**性质**：
- \\(\text{cond}(A) \geq 1\\)
- \\(\text{cond}(cA) = \text{cond}(A)\\)（\\(c \neq 0\\)）
- \\(\text{cond}(A^{-1}) = \text{cond}(A)\\)

#### 误差分析

对于线性方程组 \\(Ax = b\\)，如果右端项有扰动 \\(\Delta b\\)，则解的相对误差满足：
\\[\frac{\|\Delta x\|}{\|x\|} \leq \text{cond}(A) \frac{\|\Delta b\|}{\|b\|}\\]

**解释**：条件数放大了输入误差，条件数越大，问题越病态。

#### 迭代改善

**残差**：\\(r^{(k)} = b - Ax^{(k)}\\)

**迭代改善算法**：
1. 用单精度计算 \\(x^{(0)}\\)
2. 用双精度计算残差 \\(r^{(k)} = b - Ax^{(k)}\\)
3. 解 \\(A\delta^{(k)} = r^{(k)}\\)
4. 更新 \\(x^{(k+1)} = x^{(k)} + \delta^{(k)}\\)

## 非线性方程与方程组

### 非线性方程的数值解法

#### 二分法

**适用条件**：\\(f(a)f(b) < 0\\)，\\(f\\) 在 \\([a, b]\\) 连续

**算法**：
1. 计算 \\(c = \frac{a+b}{2}\\)
2. 如果 \\(f(a)f(c) < 0\\)，令 \\(b = c\\)；否则令 \\(a = c\\)
3. 重复直到 \\(|b - a| < \varepsilon\\)

**收敛性**：线性收敛，误差满足 \\(|x_n - x^*| \leq \frac{b-a}{2^n}\\)

#### Newton法

**公式**：
\\[x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}\\]

**几何意义**：用切线与x轴的交点作为下一个近似值。

**收敛性**：
- 局部二次收敛
- 需要 \\(f'(x^*) \neq 0\\)
- 对初值敏感

**收敛定理**：如果 \\(f(x)\\) 在根 \\(x^*\\) 附近二阶连续可导，且 \\(f'(x^*) \neq 0\\)，则当初值 \\(x_0\\) 足够接近 \\(x^*\\) 时，Newton法二次收敛。

#### 割线法

**公式**：
\\[x_{n+1} = x_n - \frac{f(x_n)(x_n - x_{n-1})}{f(x_n) - f(x_{n-1})}\\]

**特点**：
- 不需要计算导数
- 超线性收敛（收敛阶为黄金比例 \\(\frac{1+\sqrt{5}}{2} \approx 1.618\\)）
- 需要两个初值

#### 不动点迭代

将方程 \\(f(x) = 0\\) 转化为 \\(x = g(x)\\) 的形式，构造迭代：
\\[x_{n+1} = g(x_n)\\]

**收敛条件**：
- \\(|g'(x)| < 1\\) 在解的邻域内
- 初值在收敛域内

**收敛速度**：线性收敛，收敛速度由 \\(|g'(x^*)|\\) 决定

#### 加速收敛

**Aitken加速**：对于线性收敛的序列 \\(\{x_n\}\\)：
\\[\hat{x}_n = x_n - \frac{(x_{n+1} - x_n)^2}{x_{n+2} - 2x_{n+1} + x_n}\\]

**Steffensen方法**：结合不动点迭代和Aitken加速：
\\[\hat{x}_n = x_n - \frac{(g(x_n) - x_n)^2}{g(g(x_n)) - 2g(x_n) + x_n}\\]

### 非线性方程组

#### 多维Newton法

对于方程组 \\(\mathbf{F}(\mathbf{x}) = \mathbf{0}\\)，其中 \\(\mathbf{F}: \mathbb{R}^n \rightarrow \mathbb{R}^n\\)：

\\[\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \mathbf{J}(\mathbf{x}^{(k)})^{-1} \mathbf{F}(\mathbf{x}^{(k)})\\]

其中 \\(\mathbf{J}(\mathbf{x})\\) 是Jacobi矩阵：
\\[\mathbf{J}(\mathbf{x}) = \begin{pmatrix}
\frac{\partial F_1}{\partial x_1} & \frac{\partial F_1}{\partial x_2} & \cdots & \frac{\partial F_1}{\partial x_n} \\
\frac{\partial F_2}{\partial x_1} & \frac{\partial F_2}{\partial x_2} & \cdots & \frac{\partial F_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial F_n}{\partial x_1} & \frac{\partial F_n}{\partial x_2} & \cdots & \frac{\partial F_n}{\partial x_n}
\end{pmatrix}\\]

**实际计算**：
1. 解线性方程组 \\(\mathbf{J}(\mathbf{x}^{(k)}) \boldsymbol{\delta}^{(k)} = -\mathbf{F}(\mathbf{x}^{(k)})\\)
2. 更新 \\(\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \boldsymbol{\delta}^{(k)}\\)

#### 拟Newton法

**基本思想**：用近似矩阵 \\(\mathbf{B}_k\\) 代替Jacobi矩阵。

**Broyden方法**：
\\[\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{(\boldsymbol{y}_k - \mathbf{B}_k \boldsymbol{s}_k) \boldsymbol{s}_k^T}{\boldsymbol{s}_k^T \boldsymbol{s}_k}\\]

其中：
- \\(\boldsymbol{s}_k = \mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\\)
- \\(\boldsymbol{y}_k = \mathbf{F}(\mathbf{x}^{(k+1)}) - \mathbf{F}(\mathbf{x}^{(k)})\\)

**优点**：避免计算Jacobi矩阵，计算量小

## 常微分方程数值解

### 初值问题

考虑一阶常微分方程初值问题：
\\[\begin{cases}
y' = f(x, y) \\
y(x_0) = y_0
\end{cases}\\]

#### Euler方法

**显式Euler方法**：
\\[y_{n+1} = y_n + h f(x_n, y_n)\\]

**几何意义**：用切线延拓

**局部截断误差**：\\(O(h^2)\\)

**全局误差**：\\(O(h)\\)

**隐式Euler方法**：
\\[y_{n+1} = y_n + h f(x_{n+1}, y_{n+1})\\]

**特点**：需要求解非线性方程，但稳定性好

#### 改进Euler方法

**预测-校正格式**：
\\[\begin{cases}
\bar{y}_{n+1} = y_n + h f(x_n, y_n) & \text{（预测）} \\
y_{n+1} = y_n + \frac{h}{2}[f(x_n, y_n) + f(x_{n+1}, \bar{y}_{n+1})] & \text{（校正）}
\end{cases}\\]

**局部截断误差**：\\(O(h^3)\\)

#### Runge-Kutta方法

**二阶R-K方法**：
\\[\begin{cases}
k_1 = h f(x_n, y_n) \\
k_2 = h f(x_n + \frac{h}{2}, y_n + \frac{k_1}{2}) \\
y_{n+1} = y_n + k_2
\end{cases}\\]

**经典四阶R-K方法**：
\\[\begin{cases}
k_1 = h f(x_n, y_n) \\
k_2 = h f(x_n + \frac{h}{2}, y_n + \frac{k_1}{2}) \\
k_3 = h f(x_n + \frac{h}{2}, y_n + \frac{k_2}{2}) \\
k_4 = h f(x_n + h, y_n + k_3) \\
y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{cases}\\]

**局部截断误差**：\\(O(h^5)\\)

#### Adams方法

**基本思想**：用插值多项式近似 \\(f(x, y)\\)

**Adams-Bashforth公式**（显式）：
\\[y_{n+1} = y_n + h \sum_{j=0}^{k-1} \beta_j \nabla^j f_n\\]

**Adams-Moulton公式**（隐式）：
\\[y_{n+1} = y_n + h \sum_{j=0}^k \beta_j^* \nabla^j f_{n+1}\\]

其中 \\(\nabla^j f_n\\) 是后向差分算子。

**预测-校正Adams方法**：
1. 预测：用Adams-Bashforth公式
2. 校正：用Adams-Moulton公式

### 数值稳定性

#### 绝对稳定性

考虑测试方程 \\(y' = \lambda y\\)（\\(\text{Re}(\lambda) < 0\\)），数值方法的增长因子为 \\(G(h\lambda)\\)，则：

**绝对稳定条件**：\\(|G(h\lambda)| \leq 1\\)

**绝对稳定域**：\\(S = \{z \in \mathbb{C} : |G(z)| \leq 1\}\\)

**常用方法的稳定域**：
- Euler方法：\\(|1 + z| \leq 1\\)
- 改进Euler方法：\\(|1 + z + \frac{z^2}{2}| \leq 1\\)
- 四阶R-K方法：\\(|1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}| \leq 1\\)

#### A-稳定性

**定义**：如果数值方法的绝对稳定域包含整个左半平面，则称该方法是A-稳定的。

**Dahlquist第二定理**：A-稳定的线性多步方法的阶数不超过2。

#### 刚性方程

**定义**：如果方程的解包含不同时间尺度的成分，且某些成分衰减很快，则称为刚性方程。

**特点**：
- 显式方法需要很小的步长
- 需要使用隐式方法或特殊的稳定方法

### 高阶方程与方程组

#### 高阶方程

将 \\(n\\) 阶方程转化为一阶方程组：

\\[y^{(n)} = f(x, y, y', \ldots, y^{(n-1)})\\]

令 \\(u_1 = y\\)，\\(u_2 = y'\\)，\\(\ldots\\)，\\(u_n = y^{(n-1)}\\)，得到：

\\[\begin{cases}
u_1' = u_2 \\
u_2' = u_3 \\
\vdots \\
u_{n-1}' = u_n \\
u_n' = f(x, u_1, u_2, \ldots, u_n)
\end{cases}\\]

#### 一阶方程组

对于方程组：
\\[\mathbf{y}' = \mathbf{f}(x, \mathbf{y})\\]

可以直接推广单个方程的数值方法。

**向量形式的R-K方法**：
\\[\begin{cases}
\mathbf{k}_1 = h \mathbf{f}(x_n, \mathbf{y}_n) \\
\mathbf{k}_2 = h \mathbf{f}(x_n + \frac{h}{2}, \mathbf{y}_n + \frac{\mathbf{k}_1}{2}) \\
\mathbf{k}_3 = h \mathbf{f}(x_n + \frac{h}{2}, \mathbf{y}_n + \frac{\mathbf{k}_2}{2}) \\
\mathbf{k}_4 = h \mathbf{f}(x_n + h, \mathbf{y}_n + \mathbf{k}_3) \\
\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{1}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{cases}\\]

## 偏微分方程数值解

### 椭圆型方程

#### Poisson方程

考虑二维Poisson方程：
\\[\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x, y)\\]

在矩形区域 \\([0, a] \times [0, b]\\) 上，边界条件为 \\(u|_{\partial \Omega} = g\\)。

#### 有限差分方法

**网格划分**：
- \\(x_i = ih_x\\)，\\(i = 0, 1, \ldots, M\\)，\\(h_x = \frac{a}{M}\\)
- \\(y_j = jh_y\\)，\\(j = 0, 1, \ldots, N\\)，\\(h_y = \frac{b}{N}\\)

**差分格式**：
\\[\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h_x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h_y^2} = f_{i,j}\\]

**五点差分格式**（\\(h = h_x = h_y\\)）：
\\[u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} = h^2 f_{i,j}\\]

#### 迭代求解

**Jacobi迭代**：
\\[u_{i,j}^{(k+1)} = \frac{1}{4}[u_{i+1,j}^{(k)} + u_{i-1,j}^{(k)} + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k)} - h^2 f_{i,j}]\\]

**Gauss-Seidel迭代**：
\\[u_{i,j}^{(k+1)} = \frac{1}{4}[u_{i+1,j}^{(k)} + u_{i-1,j}^{(k+1)} + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k+1)} - h^2 f_{i,j}]\\]

**SOR迭代**：
\\[u_{i,j}^{(k+1)} = (1-\omega)u_{i,j}^{(k)} + \frac{\omega}{4}[u_{i+1,j}^{(k)} + u_{i-1,j}^{(k+1)} + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k+1)} - h^2 f_{i,j}]\\]

### 抛物型方程

#### 热传导方程

考虑一维热传导方程：
\\[\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}\\]

初值条件：\\(u(x, 0) = \phi(x)\\)

边界条件：\\(u(0, t) = g_1(t)\\)，\\(u(L, t) = g_2(t)\\)

#### 差分格式

**网格**：\\(x_i = ih\\)，\\(t_n = n\tau\\)

**显式格式**（前向Euler + 中心差分）：
\\[\frac{u_i^{n+1} - u_i^n}{\tau} = \alpha \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{h^2}\\]

**稳定性条件**：\\(r = \frac{\alpha \tau}{h^2} \leq \frac{1}{2}\\)

**隐式格式**（后向Euler + 中心差分）：
\\[\frac{u_i^{n+1} - u_i^n}{\tau} = \alpha \frac{u_{i+1}^{n+1} - 2u_i^{n+1} + u_{i-1}^{n+1}}{h^2}\\]

**特点**：无条件稳定，但需要解线性方程组

**Crank-Nicolson格式**：
\\[\frac{u_i^{n+1} - u_i^n}{\tau} = \frac{\alpha}{2}\left[\frac{u_{i+1}^{n+1} - 2u_i^{n+1} + u_{i-1}^{n+1}}{h^2} + \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{h^2}\right]\\]

**特点**：二阶精度，无条件稳定

### 双曲型方程

#### 波动方程

考虑一维波动方程：
\\[\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}\\]

初值条件：
\\[u(x, 0) = \phi(x), \quad \frac{\partial u}{\partial t}(x, 0) = \psi(x)\\]

#### 差分格式

**显式格式**：
\\[\frac{u_i^{n+1} - 2u_i^n + u_i^{n-1}}{\tau^2} = c^2 \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{h^2}\\]

**稳定性条件**（CFL条件）：\\(\frac{c\tau}{h} \leq 1\\)

**物理意义**：数值区域必须包含物理影响区域

## 数值分析在建模中的应用

### 计算流体力学

#### Navier-Stokes方程

不可压缩流动的Navier-Stokes方程：
\\[\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{v}\\]
\\[\nabla \cdot \mathbf{v} = 0\\]

其中 \\(\mathbf{v}\\) 是速度，\\(p\\) 是压力，\\(\rho\\) 是密度，\\(\nu\\) 是动力粘度。

#### 有限体积方法

**基本思想**：将计算域分成控制体积，在每个控制体积上满足守恒定律。

**对流项离散**：
- **迎风格式**：数值粘性大，稳定性好
- **中心差分**：精度高，但可能出现振荡
- **混合格式**：结合迎风和中心差分的优点

#### SIMPLE算法

**压力-速度耦合算法**：

1. 假设压力场 \\(p^*\\)
2. 求解动量方程得到速度 \\(\mathbf{v}^*\\)
3. 计算速度修正 \\(\mathbf{v}'\\)
4. 求解压力修正方程得到 \\(p'\\)
5. 更新压力和速度
6. 重复直到收敛

### 结构力学

#### 有限元方法

**基本思想**：将连续的计算域离散为有限个单元，在每个单元上用简单函数近似解。

**变分原理**：将微分方程转化为变分问题，寻找使泛函极值的函数。

**例子：一维杆的轴向变形**

控制方程：\\(-\frac{d}{dx}\left(EA\frac{du}{dx}\right) = f(x)\\)

变分形式：\\(\min \int_0^L \left[\frac{1}{2}EA\left(\frac{du}{dx}\right)^2 - fu\right] dx\\)

**单元分析**：
1. 选择形函数（如线性函数）
2. 建立单元刚度矩阵
3. 组装总体刚度矩阵
4. 施加边界条件
5. 求解线性方程组

### 优化问题

#### 无约束优化

**梯度下降法**：
\\[\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \alpha_k \nabla f(\mathbf{x}^{(k)})\\]

**Newton方法**：
\\[\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - [\nabla^2 f(\mathbf{x}^{(k)})]^{-1} \nabla f(\mathbf{x}^{(k)})\\]

**拟Newton方法**（BFGS）：
\\[\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \alpha_k \mathbf{H}_k \nabla f(\mathbf{x}^{(k)})\\]

其中 \\(\mathbf{H}_k\\) 是Hessian矩阵的近似。

#### 约束优化

**拉格朗日乘数法**：对于问题
\\[\min f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) = 0\\]

构造拉格朗日函数：
\\[L(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x})\\]

**KKT条件**：对于不等式约束问题，最优解必须满足Karush-Kuhn-Tucker条件。

### 数据拟合与机器学习

#### 最小二乘拟合

对于线性模型 \\(y = \sum_{j=0}^n a_j \phi_j(x)\\)，最小化：
\\[S = \sum_{i=1}^m \left[y_i - \sum_{j=0}^n a_j \phi_j(x_i)\right]^2\\]

**正规方程**：\\(\mathbf{G}\mathbf{a} = \mathbf{d}\\)

其中 \\(G_{ij} = \sum_{k=1}^m \phi_i(x_k)\phi_j(x_k)\\)，\\(d_i = \sum_{k=1}^m y_k\phi_i(x_k)\\)

#### 神经网络训练

**反向传播算法**：

1. 前向传播计算输出
2. 计算输出层误差
3. 反向传播误差到隐藏层
4. 更新权重和偏置

**梯度下降更新**：
\\[w_{ij}^{(k+1)} = w_{ij}^{(k)} - \eta \frac{\partial E}{\partial w_{ij}}\\]

其中 \\(\eta\\) 是学习率，\\(E\\) 是误差函数。

## 高性能计算

### 并行算法

#### 并行矩阵运算

**矩阵乘法的并行化**：
- **行分块**：每个处理器负责部分行
- **列分块**：每个处理器负责部分列
- **块分块**：将矩阵分成子块

**通信开销**：需要在处理器间交换数据

#### 并行求解线性方程组

**并行Gauss消元**：
- 每个处理器负责部分行
- 需要在消元过程中同步

**并行迭代方法**：
- Jacobi方法易于并行化
- Gauss-Seidel需要特殊处理

### 加速技术

#### 向量化

**SIMD指令**：单指令多数据，可以同时处理多个数据。

**向量化循环**：
```c
// 串行代码
for (i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}

// 向量化代码
#pragma vector
for (i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
```

#### 缓存优化

**局部性原理**：
- **时间局部性**：最近访问的数据很可能再次被访问
- **空间局部性**：邻近的数据很可能被访问

**循环优化技术**：
- **循环交换**：改变循环顺序以提高缓存命中率
- **循环分块**：将大循环分成小块
- **循环展开**：减少循环开销

## 小结

数值分析为数学建模提供了强大的计算工具：

1. **误差控制**：理解和控制计算过程中的各种误差
2. **算法选择**：根据问题特点选择合适的数值方法
3. **稳定性分析**：确保算法的数值稳定性
4. **效率优化**：通过并行化和优化技术提高计算效率
5. **实际应用**：解决科学和工程中的实际问题

掌握数值分析的关键在于：
- 理解各种数值方法的原理和适用范围
- 掌握误差分析和稳定性理论
- 能够编程实现数值算法
- 了解现代高性能计算技术
- 能够解决实际建模中的计算问题

数值分析是连接数学理论与实际应用的桥梁，是现代科学计算和工程仿真的基础。
