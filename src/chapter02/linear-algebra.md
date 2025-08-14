# 线性代数

> "线性代数是数学的基础，它为我们提供了理解和操作多维空间的语言。" 
> —— 数学家吉尔伯特·斯特朗

线性代数是数学建模中最基础也是最重要的工具之一。它不仅提供了处理多元线性关系的数学框架，更是现代数据科学、机器学习和工程计算的核心基础。

## 向量空间理论

### 向量的概念与运算

#### 向量的定义

向量是具有大小和方向的量，在n维空间中可以表示为：
\\[\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}\\]

#### 向量运算

**加法**：
\\[\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}\\]

**数乘**：
\\[c\mathbf{v} = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{pmatrix}\\]

**内积（点积）**：
\\[\mathbf{u} \cdot \mathbf{v} = u_1v_1 + u_2v_2 + \cdots + u_nv_n = \sum_{i=1}^n u_iv_i\\]

**外积（叉积，仅适用于三维）**：
\\[\mathbf{u} \times \mathbf{v} = \begin{pmatrix} u_2v_3 - u_3v_2 \\ u_3v_1 - u_1v_3 \\ u_1v_2 - u_2v_1 \end{pmatrix}\\]

#### 向量的几何意义

**模长（范数）**：
\\[\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}\\]

**夹角**：
\\[\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\\]

**正交性**：两向量正交当且仅当 \\(\mathbf{u} \cdot \mathbf{v} = 0\\)

### 向量空间的公理

向量空间 \\(V\\) 是满足以下8个公理的集合：

1. **加法封闭性**：\\(\mathbf{u}, \mathbf{v} \in V \Rightarrow \mathbf{u} + \mathbf{v} \in V\\)
2. **加法交换律**：\\(\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}\\)
3. **加法结合律**：\\((\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})\\)
4. **零向量存在**：存在 \\(\mathbf{0} \in V\\) 使得 \\(\mathbf{v} + \mathbf{0} = \mathbf{v}\\)
5. **负向量存在**：对任意 \\(\mathbf{v} \in V\\)，存在 \\(-\mathbf{v} \in V\\) 使得 \\(\mathbf{v} + (-\mathbf{v}) = \mathbf{0}\\)
6. **数乘封闭性**：\\(c \in \mathbb{R}, \mathbf{v} \in V \Rightarrow c\mathbf{v} \in V\\)
7. **数乘分配律**：\\(c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}\\) 和 \\((c + d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}\\)
8. **数乘结合律**：\\((cd)\mathbf{v} = c(d\mathbf{v})\\) 和 \\(1\mathbf{v} = \mathbf{v}\\)

### 线性相关性与基

#### 线性组合

向量 \\(\mathbf{v}\\) 是向量组 \\(\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}\\) 的线性组合，如果存在标量 \\(c_1, c_2, \ldots, c_k\\) 使得：
\\[\mathbf{v} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k\\]

#### 线性相关性

向量组 \\(\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}\\) 线性相关，如果存在不全为零的标量 \\(c_1, c_2, \ldots, c_k\\) 使得：
\\[c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}\\]

否则称这些向量线性无关。

#### 基与维数

**基**：向量空间 \\(V\\) 的一组基是 \\(V\\) 中线性无关且能生成整个空间的向量组。

**维数**：向量空间的维数等于其任意一组基中向量的个数。

**标准基**：\\(\mathbb{R}^n\\) 的标准基是：
\\[\mathbf{e}_1 = \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \mathbf{e}_2 = \begin{pmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{pmatrix}, \ldots, \mathbf{e}_n = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{pmatrix}\\]

### 子空间

#### 子空间的定义

集合 \\(W \subseteq V\\) 是向量空间 \\(V\\) 的子空间，如果：
1. \\(\mathbf{0} \in W\\)
2. 对任意 \\(\mathbf{u}, \mathbf{v} \in W\\)，有 \\(\mathbf{u} + \mathbf{v} \in W\\)
3. 对任意 \\(c \in \mathbb{R}, \mathbf{v} \in W\\)，有 \\(c\mathbf{v} \in W\\)

#### 重要的子空间

**列空间**：矩阵 \\(A\\) 的列空间 \\(\text{Col}(A)\\) 是由其列向量生成的子空间。

**零空间**：矩阵 \\(A\\) 的零空间 \\(\text{Null}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}\\)。

**行空间**：矩阵 \\(A\\) 的行空间 \\(\text{Row}(A)\\) 是由其行向量生成的子空间。

**左零空间**：矩阵 \\(A\\) 的左零空间 \\(\text{Null}(A^T) = \{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\}\\)。

## 矩阵理论

### 矩阵的基本概念

#### 矩阵定义

\\(m \times n\\) 矩阵是一个矩形数组：
\\[A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}\\]

#### 特殊矩阵

**方阵**：行数等于列数的矩阵。

**对角矩阵**：只有对角元素非零的方阵：
\\[D = \begin{pmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{pmatrix}\\]

**单位矩阵**：对角元素全为1的对角矩阵：
\\[I = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}\\]

**对称矩阵**：满足 \\(A = A^T\\) 的方阵。

**反对称矩阵**：满足 \\(A = -A^T\\) 的方阵。

**正交矩阵**：满足 \\(Q^TQ = I\\) 的方阵。

### 矩阵运算

#### 基本运算

**加法**：
\\[(A + B)_{ij} = a_{ij} + b_{ij}\\]

**数乘**：
\\[(cA)_{ij} = ca_{ij}\\]

**乘法**：
\\[(AB)_{ij} = \sum_{k=1}^p a_{ik}b_{kj}\\]

**转置**：
\\[(A^T)_{ij} = a_{ji}\\]

#### 矩阵乘法的性质

1. **结合律**：\\((AB)C = A(BC)\\)
2. **分配律**：\\(A(B + C) = AB + AC\\)
3. **转置性质**：\\((AB)^T = B^TA^T\\)
4. **一般不满足交换律**：\\(AB \neq BA\\)

### 矩阵的逆

#### 可逆矩阵

方阵 \\(A\\) 可逆，如果存在矩阵 \\(A^{-1}\\) 使得：
\\[AA^{-1} = A^{-1}A = I\\]

#### 可逆的条件

矩阵 \\(A\\) 可逆当且仅当：
- \\(\det(A) \neq 0\\)
- \\(A\\) 的列向量线性无关
- \\(A\\) 的零空间只包含零向量
- \\(A\\) 的列空间是整个 \\(\mathbb{R}^n\\)

#### 逆矩阵的计算

**2×2矩阵**：
\\[A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \quad A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}\\]

**一般方法**：高斯-约旦消元法
\\[[A|I] \rightarrow [I|A^{-1}]\\]

### 矩阵的秩

#### 秩的定义

矩阵的秩是其线性无关行（或列）的最大个数，记作 \\(\text{rank}(A)\\)。

#### 秩的性质

1. \\(\text{rank}(A) = \text{rank}(A^T)\\)
2. \\(\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))\\)
3. \\(\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)\\)
4. 对可逆矩阵 \\(P, Q\\)：\\(\text{rank}(PAQ) = \text{rank}(A)\\)

#### 满秩矩阵

- **行满秩**：\\(\text{rank}(A) = m\\)（行数）
- **列满秩**：\\(\text{rank}(A) = n\\)（列数）
- **满秩**：\\(\text{rank}(A) = \min(m, n)\\)

## 线性方程组

### 线性方程组的矩阵表示

线性方程组：
\\[\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}\\]

矩阵形式：\\(A\mathbf{x} = \mathbf{b}\\)

增广矩阵：\\([A|\mathbf{b}]\\)

### 解的存在性和唯一性

根据克拉默法则和矩阵理论：

1. **有唯一解**：\\(\text{rank}(A) = \text{rank}([A|\mathbf{b}]) = n\\)
2. **有无穷解**：\\(\text{rank}(A) = \text{rank}([A|\mathbf{b}]) < n\\)
3. **无解**：\\(\text{rank}(A) < \text{rank}([A|\mathbf{b}])\\)

### 齐次线性方程组

方程组 \\(A\mathbf{x} = \mathbf{0}\\) 总有解（至少有零解）。

**基础解系**：齐次方程组解空间的一组基。

如果 \\(\text{rank}(A) = r < n\\)，则基础解系包含 \\(n - r\\) 个线性无关的解向量。

### 非齐次线性方程组

通解结构：
\\[\mathbf{x} = \mathbf{x}_0 + \mathbf{x}_h\\]

其中 \\(\mathbf{x}_0\\) 是特解，\\(\mathbf{x}_h\\) 是齐次方程组的通解。

### 高斯消元法

通过行变换将增广矩阵化为阶梯形：

**行阶梯形矩阵**：
1. 非零行在零行之上
2. 每行的首个非零元素（主元）在上一行主元的右边

**最简行阶梯形矩阵**：
1. 满足行阶梯形的条件
2. 主元为1
3. 主元所在列的其他元素为0

## 特征值与特征向量

### 基本概念

#### 特征值和特征向量的定义

对于 \\(n \times n\\) 矩阵 \\(A\\)，如果存在非零向量 \\(\mathbf{v}\\) 和标量 \\(\lambda\\) 使得：
\\[A\mathbf{v} = \lambda\mathbf{v}\\]

则称 \\(\lambda\\) 为 \\(A\\) 的特征值，\\(\mathbf{v}\\) 为对应的特征向量。

#### 特征多项式

特征值是特征方程的根：
\\[\det(A - \lambda I) = 0\\]

\\(\det(A - \lambda I)\\) 称为特征多项式。

#### 特征空间

对应特征值 \\(\lambda\\) 的特征空间是：
\\[E_\lambda = \{\mathbf{v} : A\mathbf{v} = \lambda\mathbf{v}\} = \text{Null}(A - \lambda I)\\]

### 特征值的性质

1. **迹**：\\(\text{tr}(A) = \lambda_1 + \lambda_2 + \cdots + \lambda_n\\)
2. **行列式**：\\(\det(A) = \lambda_1 \lambda_2 \cdots \lambda_n\\)
3. **相似不变性**：相似矩阵有相同的特征值
4. **实对称矩阵的特征值都是实数**

### 对角化

#### 可对角化的条件

矩阵 \\(A\\) 可对角化当且仅当 \\(A\\) 有 \\(n\\) 个线性无关的特征向量。

如果 \\(A\\) 可对角化，则存在可逆矩阵 \\(P\\) 使得：
\\[P^{-1}AP = D\\]

其中 \\(D\\) 是对角矩阵，\\(P\\) 的列是 \\(A\\) 的特征向量。

#### 对称矩阵的对角化

**谱定理**：任何实对称矩阵都可以正交对角化，即存在正交矩阵 \\(Q\\) 使得：
\\[Q^TAQ = D\\]

### 二次型

#### 二次型的定义

\\(n\\) 元二次型是形如：
\\[f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^n \sum_{j=1}^n a_{ij}x_ix_j = \mathbf{x}^TA\mathbf{x}\\]

的函数，其中 \\(A\\) 是对称矩阵。

#### 二次型的分类

根据特征值的符号：
- **正定**：所有特征值 > 0
- **负定**：所有特征值 < 0
- **半正定**：所有特征值 ≥ 0
- **半负定**：所有特征值 ≤ 0
- **不定**：既有正特征值又有负特征值

#### 主轴定理

通过正交变换 \\(\mathbf{x} = Q\mathbf{y}\\)，二次型可化为标准形：
\\[f = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2\\]

## 矩阵分解

### LU分解

将矩阵分解为下三角矩阵和上三角矩阵的乘积：
\\[A = LU\\]

其中 \\(L\\) 是下三角矩阵，\\(U\\) 是上三角矩阵。

**PLU分解**（带行交换）：
\\[PA = LU\\]

### QR分解

将矩阵分解为正交矩阵和上三角矩阵的乘积：
\\[A = QR\\]

其中 \\(Q\\) 是正交矩阵，\\(R\\) 是上三角矩阵。

#### Gram-Schmidt正交化

给定线性无关向量组 \\(\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}\\)，可构造正交向量组：

\\[\mathbf{u}_1 = \mathbf{v}_1\\]

\\[\mathbf{u}_2 = \mathbf{v}_2 - \frac{\mathbf{v}_2 \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1}\mathbf{u}_1\\]

\\[\mathbf{u}_3 = \mathbf{v}_3 - \frac{\mathbf{v}_3 \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1}\mathbf{u}_1 - \frac{\mathbf{v}_3 \cdot \mathbf{u}_2}{\mathbf{u}_2 \cdot \mathbf{u}_2}\mathbf{u}_2\\]

以此类推。

### 奇异值分解（SVD）

任何 \\(m \times n\\) 矩阵 \\(A\\) 都可以分解为：
\\[A = U\Sigma V^T\\]

其中：
- \\(U\\) 是 \\(m \times m\\) 正交矩阵
- \\(V\\) 是 \\(n \times n\\) 正交矩阵  
- \\(\Sigma\\) 是 \\(m \times n\\) 对角矩阵，对角元素 \\(\sigma_i \geq 0\\) 称为奇异值

#### SVD的几何意义

SVD 描述了线性变换的完整几何结构：
1. \\(V^T\\)：坐标系旋转
2. \\(\Sigma\\)：沿坐标轴缩放
3. \\(U\\)：坐标系旋转

#### SVD的应用

- **数据压缩**：保留主要奇异值
- **主成分分析**：降维
- **矩阵的伪逆**：\\(A^+ = V\Sigma^+U^T\\)
- **最小二乘问题**

## 线性变换

### 线性变换的定义

函数 \\(T: V \rightarrow W\\) 是线性变换，如果：
1. \\(T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})\\)
2. \\(T(c\mathbf{v}) = cT(\mathbf{v})\\)

### 线性变换的矩阵表示

选定基后，线性变换可用矩阵表示：
\\[T(\mathbf{x}) = A\mathbf{x}\\]

### 重要的线性变换

#### 几何变换

**旋转变换**（二维，逆时针旋转角度 \\(\theta\\)）：
\\[R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\\]

**反射变换**（关于x轴）：
\\[S = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\\]

**缩放变换**：
\\[D = \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}\\]

**剪切变换**：
\\[H = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}\\]

#### 投影变换

**正交投影**到子空间 \\(W\\)：
\\[P = A(A^TA)^{-1}A^T\\]

其中 \\(A\\) 的列向量构成 \\(W\\) 的基。

### 核与像

**核（零空间）**：
\\[\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}\\]

**像（值域）**：
\\[\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}\\]

**维数定理**：
\\[\dim(V) = \dim(\ker(T)) + \dim(\text{Im}(T))\\]

## 线性代数在建模中的应用

### 线性回归

#### 最小二乘法

对于线性模型 \\(\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\epsilon}\\)，最小二乘解为：
\\[\hat{\boldsymbol{\beta}} = (X^TX)^{-1}X^T\mathbf{y}\\]

#### 正规方程

最小二乘问题等价于求解正规方程：
\\[X^TX\boldsymbol{\beta} = X^T\mathbf{y}\\]

#### 几何解释

最小二乘解是 \\(\mathbf{y}\\) 在列空间 \\(\text{Col}(X)\\) 上的正交投影。

### 主成分分析（PCA）

#### 问题描述

给定数据矩阵 \\(X\\)，寻找低维表示保留最大方差。

#### 数学表述

1. 计算协方差矩阵：\\(C = \frac{1}{n-1}X^TX\\)
2. 求特征值分解：\\(C = PDP^T\\)
3. 选择前 \\(k\\) 个主成分对应的特征向量

#### PCA的应用

- **降维**：减少数据维度
- **数据可视化**：高维数据的二维展示
- **噪声去除**：保留主要成分，去除噪声
- **特征提取**：提取数据的主要特征

### 马尔可夫链

#### 转移矩阵

马尔可夫链的状态转移由转移矩阵 \\(P\\) 描述：
\\[P_{ij} = P(X_{n+1} = j | X_n = i)\\]

#### 性质

- 每行元素和为1：\\(\sum_j P_{ij} = 1\\)
- \\(n\\) 步转移概率：\\(P^{(n)} = P^n\\)

#### 平稳分布

平稳分布 \\(\boldsymbol{\pi}\\) 满足：
\\[\boldsymbol{\pi}^T P = \boldsymbol{\pi}^T\\]

即 \\(\boldsymbol{\pi}\\) 是转移矩阵 \\(P^T\\) 对应特征值1的特征向量。

### 网络分析

#### 图的邻接矩阵

对于 \\(n\\) 个节点的图，邻接矩阵 \\(A\\) 定义为：
\\[A_{ij} = \begin{cases} 1 & \text{如果节点 } i \text{ 和 } j \text{ 相连} \\ 0 & \text{否则} \end{cases}\\]

#### 度矩阵

度矩阵 \\(D\\) 是对角矩阵：
\\[D_{ii} = \sum_j A_{ij}\\]

#### 拉普拉斯矩阵

\\[L = D - A\\]

拉普拉斯矩阵的特征值提供了图的重要信息：
- 第二小特征值（Fiedler值）：连通性度量
- 特征向量：图的分割

#### PageRank算法

Google的PageRank算法基于特征向量：
\\[(1-d)A + \frac{d}{n}\mathbf{1}\mathbf{1}^T)\mathbf{r} = \mathbf{r}\\]

其中 \\(\mathbf{r}\\) 是PageRank向量，\\(d\\) 是阻尼系数。

### 线性规划

#### 标准形式

\\[\min \mathbf{c}^T\mathbf{x}\\]
\\[\text{s.t. } A\mathbf{x} = \mathbf{b}, \mathbf{x} \geq \mathbf{0}\\]

#### 单纯形法

单纯形法在可行域的顶点间移动寻找最优解，每个顶点对应基本可行解。

#### 对偶理论

原问题：\\(\min \mathbf{c}^T\mathbf{x}\\)，\\(A\mathbf{x} = \mathbf{b}\\)，\\(\mathbf{x} \geq \mathbf{0}\\)

对偶问题：\\(\max \mathbf{b}^T\mathbf{y}\\)，\\(A^T\mathbf{y} \leq \mathbf{c}\\)

强对偶定理保证最优值相等。

### 控制理论

#### 状态空间模型

线性系统的状态空间表示：
\\[\frac{d\mathbf{x}}{dt} = A\mathbf{x} + B\mathbf{u}\\]
\\[\mathbf{y} = C\mathbf{x} + D\mathbf{u}\\]

#### 能控性

系统能控当且仅当能控性矩阵满秩：
\\[\mathcal{C} = [B, AB, A^2B, \ldots, A^{n-1}B]\\]

#### 能观性

系统能观当且仅当能观性矩阵满秩：
\\[\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix}\\]

#### 系统稳定性

线性系统稳定当且仅当矩阵 \\(A\\) 的所有特征值实部为负。

### 图像处理

#### 图像的矩阵表示

灰度图像可表示为矩阵，每个元素是像素强度。

#### 图像压缩

使用SVD进行图像压缩：
\\[A = U\Sigma V^T \approx U_k\Sigma_k V_k^T\\]

保留前 \\(k\\) 个奇异值可实现压缩。

#### 图像变换

- **平移**：\\(\mathbf{x}' = \mathbf{x} + \mathbf{t}\\)
- **旋转**：\\(\mathbf{x}' = R\mathbf{x}\\)
- **缩放**：\\(\mathbf{x}' = S\mathbf{x}\\)

齐次坐标系统一处理：
\\[\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{pmatrix}\begin{pmatrix} x \\ y \\ 1 \end{pmatrix}\\]

## 数值线性代数

### 矩阵范数

#### 向量范数

- **1-范数**：\\(\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|\\)
- **2-范数**：\\(\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}\\)
- **∞-范数**：\\(\|\mathbf{x}\|_\infty = \max_{1 \leq i \leq n} |x_i|\\)

#### 矩阵范数

- **Frobenius范数**：\\(\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}\\)
- **谱范数**：\\(\|A\|_2 = \sigma_{\max}(A)\\)（最大奇异值）
- **1-范数**：\\(\|A\|_1 = \max_j \sum_i |a_{ij}|\\)（最大列和）
- **∞-范数**：\\(\|A\|_\infty = \max_i \sum_j |a_{ij}|\\)（最大行和）

### 条件数

矩阵 \\(A\\) 的条件数定义为：
\\[\kappa(A) = \|A\|\|A^{-1}\|\\]

条件数度量了矩阵的数值稳定性：
- \\(\kappa(A) = 1\\)：最好的条件（正交矩阵）
- \\(\kappa(A)\\) 很大：病态矩阵，数值不稳定

### 迭代方法

#### Jacobi迭代

对于方程组 \\(A\mathbf{x} = \mathbf{b}\\)，Jacobi迭代为：
\\[x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij}x_j^{(k)}\right)\\]

#### Gauss-Seidel迭代

\\[x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j < i} a_{ij}x_j^{(k+1)} - \sum_{j > i} a_{ij}x_j^{(k)}\right)\\]

#### 收敛性

迭代方法收敛的充分条件：
- **严格对角占优**：\\(|a_{ii}| > \sum_{j \neq i} |a_{ij}|\\)
- **正定矩阵**：对称正定矩阵保证收敛

### 最小二乘问题的数值解法

对于超定方程组 \\(A\mathbf{x} = \mathbf{b}\\)（\\(m > n\\)），有几种数值方法：

#### 正规方程法

解 \\(A^TA\mathbf{x} = A^T\mathbf{b}\\)

**缺点**：条件数平方，数值不稳定。

#### QR分解法

\\(A = QR\\)，则 \\(\mathbf{x} = R^{-1}Q^T\mathbf{b}\\)

**优点**：数值稳定。

#### SVD法

\\(A = U\Sigma V^T\\)，则 \\(\mathbf{x} = V\Sigma^+U^T\mathbf{b}\\)

**优点**：最稳定，可处理秩亏情况。

## 矩阵微积分

### 向量函数的导数

#### 标量对向量的导数

\\[\frac{\partial f}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}\\]

#### 向量对标量的导数

\\[\frac{d\mathbf{f}}{dt} = \begin{pmatrix} \frac{df_1}{dt} \\ \frac{df_2}{dt} \\ \vdots \\ \frac{df_n}{dt} \end{pmatrix}\\]

#### 常用公式

- \\(\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T\mathbf{x}) = \mathbf{a}\\)
- \\(\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}\\)
- \\(\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T\mathbf{x}) = 2\mathbf{x}\\)

### 矩阵函数的导数

#### 矩阵指数

\\[e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots\\]

**性质**：
- \\(\frac{d}{dt}e^{At} = Ae^{At}\\)
- 如果 \\(AB = BA\\)，则 \\(e^{A+B} = e^A e^B\\)

#### 应用

线性微分方程组 \\(\frac{d\mathbf{x}}{dt} = A\mathbf{x}\\) 的解：
\\[\mathbf{x}(t) = e^{At}\mathbf{x}(0)\\]

## 小结

线性代数为数学建模提供了强大的工具：

1. **多元线性关系**：向量和矩阵描述多变量系统
2. **几何直觉**：线性变换的几何意义
3. **数值计算**：高效的算法和稳定性分析
4. **数据分析**：主成分分析、回归分析等
5. **系统分析**：状态空间模型、稳定性分析
6. **优化问题**：线性规划、二次规划

掌握线性代数的关键在于：
- 理解抽象概念的几何意义
- 熟练掌握矩阵运算和分解
- 了解数值稳定性和算法效率
- 能够将实际问题转化为线性代数问题

线性代数是现代数学建模的基石，与微积分、概率统计等工具结合，构成了解决复杂问题的完整框架。
