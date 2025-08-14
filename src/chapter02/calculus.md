# 微积分基础

> "微积分是数学的主要工具，而数学是科学的主要工具。" 
> —— 数学家理查德·费曼

微积分作为数学分析的核心，是数学建模中最重要的工具之一。它提供了描述连续变化和累积效应的数学语言，使我们能够精确地分析动态系统、优化问题和变化过程。

## 微分学：变化的数学

### 导数的本质与意义

#### 导数定义

函数 \\(f(x)\\) 在点 \\(x_0\\) 处的导数定义为：

\\[f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}\\]

这个定义蕴含着深刻的数学思想：
- **局部线性化**：将复杂的非线性函数在小范围内用线性函数近似
- **瞬时变化率**：描述函数在某一点的即时变化速度
- **几何意义**：函数图像在该点的切线斜率

#### 导数的几何解释

**切线方程**

函数 \\(y = f(x)\\) 在点 \\((x_0, f(x_0))\\) 处的切线方程为：
\\[y - f(x_0) = f'(x_0)(x - x_0)\\]

**线性逼近**

在点 \\(x_0\\) 附近，函数可以用其切线近似：
\\[f(x) \approx f(x_0) + f'(x_0)(x - x_0)\\]

这种线性逼近是数学建模中简化复杂关系的重要手段。

### 高阶导数与泰勒展开

#### 高阶导数的意义

- **二阶导数** \\(f''(x)\\)：描述函数变化率的变化率，反映函数的凹凸性
- **三阶导数** \\(f'''(x)\\)：描述凹凸性的变化
- **n阶导数** \\(f^{(n)}(x)\\)：描述更高层次的变化特征

#### 泰勒定理

**泰勒公式**

函数 \\(f(x)\\) 在点 \\(x_0\\) 处的n阶泰勒展开：

\\[f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x-x_0)^n + R_n(x)\\]

其中 \\(R_n(x)\\) 是余项。

**麦克劳林级数**

当 \\(x_0 = 0\\) 时的特殊情况：
\\[f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \cdots\\]

**常用函数的泰勒展开**

\\[e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots\\]

\\[\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots\\]

\\[\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots\\]

\\[\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots, \quad |x| < 1\\]

### 多元函数微分学

#### 偏导数

对于多元函数 \\(f(x, y)\\)，偏导数定义为：

\\[\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}\\]

\\[\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y+h) - f(x, y)}{h}\\]

#### 梯度向量

\\[\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)\\]

梯度向量指向函数增长最快的方向，其模长表示最大变化率。

#### 方向导数

函数 \\(f(x, y)\\) 在点 \\((x_0, y_0)\\) 沿单位向量 \\(\mathbf{u} = (u_1, u_2)\\) 方向的方向导数：

\\[D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \frac{\partial f}{\partial x}u_1 + \frac{\partial f}{\partial y}u_2\\]

#### 全微分

\\[df = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy\\]

全微分描述了函数在各个方向上的线性近似。

### 微分在建模中的应用

#### 变化率建模

**人口动力学**

Malthus模型：\\(\frac{dP}{dt} = rP\\)
- 解：\\(P(t) = P_0 e^{rt}\\)
- 特点：指数增长

Logistic模型：\\(\frac{dP}{dt} = rP(1 - \frac{P}{K})\\)
- K为环境容量
- 描述有限环境中的人口增长

**化学反应动力学**

一级反应：\\(\frac{dc}{dt} = -kc\\)
- 解：\\(c(t) = c_0 e^{-kt}\\)
- 特点：指数衰减

二级反应：\\(\frac{dc}{dt} = -kc^2\\)
- 解：\\(\frac{1}{c(t)} = \frac{1}{c_0} + kt\\)

**传热模型**

牛顿冷却定律：\\(\frac{dT}{dt} = -k(T - T_{\text{env}})\\)
- \\(T\\)：物体温度
- \\(T_{\text{env}}\\)：环境温度
- 解：\\(T(t) = T_{\text{env}} + (T_0 - T_{\text{env}})e^{-kt}\\)

#### 优化问题

**单变量优化**

寻找函数 \\(f(x)\\) 的极值：
1. 求解 \\(f'(x) = 0\\) 得到临界点
2. 用二阶导数判断极值性质：
   - \\(f''(x) > 0\\)：极小值
   - \\(f''(x) < 0\\)：极大值

**多变量优化**

寻找函数 \\(f(x, y)\\) 的极值：
1. 求解方程组：\\(\frac{\partial f}{\partial x} = 0\\)，\\(\frac{\partial f}{\partial y} = 0\\)
2. 用Hessian矩阵判断：
   \\[H = \begin{pmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{pmatrix}\\]
   
   - \\(\det(H) > 0, \frac{\partial^2 f}{\partial x^2} > 0\\)：极小值
   - \\(\det(H) > 0, \frac{\partial^2 f}{\partial x^2} < 0\\)：极大值
   - \\(\det(H) < 0\\)：鞍点

**拉格朗日乘数法**

约束优化问题：
\\[\min f(x, y) \quad \text{s.t.} \quad g(x, y) = 0\\]

构造拉格朗日函数：
\\[L(x, y, \lambda) = f(x, y) + \lambda g(x, y)\\]

必要条件：
\\[\frac{\partial L}{\partial x} = 0, \quad \frac{\partial L}{\partial y} = 0, \quad \frac{\partial L}{\partial \lambda} = 0\\]

#### 敏感性分析

研究参数变化对结果的影响。对于函数 \\(y = f(x, p)\\)，其中 \\(p\\) 是参数：

**绝对敏感性**：\\(S_{\text{abs}} = \frac{\partial f}{\partial p}\\)

**相对敏感性**：\\(S_{\text{rel}} = \frac{\partial f}{\partial p} \cdot \frac{p}{f}\\)

敏感性分析帮助识别模型中的关键参数，指导实验设计和数据收集。

## 积分学：累积的数学

### 定积分的本质

#### 定积分定义

函数 \\(f(x)\\) 在区间 \\([a, b]\\) 上的定积分：

\\[\int_a^b f(x) dx = \lim_{n \to \infty} \sum_{i=1}^n f(\xi_i) \Delta x_i\\]

其中 \\(\Delta x_i = \frac{b-a}{n}\\)，\\(\xi_i \in [x_{i-1}, x_i]\\)。

#### 几何意义

定积分表示函数 \\(f(x)\\) 与x轴之间的有向面积。

#### 物理意义

- **位移**：速度函数的积分
- **功**：力函数沿路径的积分
- **质量**：密度函数在区域上的积分

### 积分计算技巧

#### 基本积分公式

\\[\int x^n dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)\\]

\\[\int \frac{1}{x} dx = \ln|x| + C\\]

\\[\int e^x dx = e^x + C\\]

\\[\int \sin x dx = -\cos x + C\\]

\\[\int \cos x dx = \sin x + C\\]

#### 换元积分法

**第一类换元**（凑微分）
\\[\int f(g(x))g'(x) dx = \int f(u) du \quad (u = g(x))\\]

**第二类换元**（变量替换）
\\[\int f(x) dx = \int f(\phi(t))\phi'(t) dt \quad (x = \phi(t))\\]

#### 分部积分法

\\[\int u dv = uv - \int v du\\]

选择原则：LIATE（对数、反三角、代数、三角、指数）

#### 有理函数积分

通过部分分式分解：
\\[\frac{P(x)}{Q(x)} = \sum \frac{A_i}{(x-a_i)^{n_i}} + \sum \frac{B_i x + C_i}{(x^2 + p_i x + q_i)^{m_i}}\\]

### 反常积分

#### 无穷限积分

\\[\int_a^{+\infty} f(x) dx = \lim_{t \to +\infty} \int_a^t f(x) dx\\]

**收敛性判断**

比较判别法：设 \\(0 \leq f(x) \leq g(x)\\)
- 若 \\(\int_a^{+\infty} g(x) dx\\) 收敛，则 \\(\int_a^{+\infty} f(x) dx\\) 收敛
- 若 \\(\int_a^{+\infty} f(x) dx\\) 发散，则 \\(\int_a^{+\infty} g(x) dx\\) 发散

#### 无界函数积分

对于在点 \\(c \in [a, b]\\) 处无界的函数：
\\[\int_a^b f(x) dx = \lim_{\epsilon \to 0^+} \left[\int_a^{c-\epsilon} f(x) dx + \int_{c+\epsilon}^b f(x) dx\right]\\]

### 重积分

#### 二重积分

**直角坐标系**
\\[\iint_D f(x, y) dA = \int_a^b \int_{y_1(x)}^{y_2(x)} f(x, y) dy dx\\]

**极坐标系**
\\[\iint_D f(x, y) dA = \int_{\alpha}^{\beta} \int_{r_1(\theta)}^{r_2(\theta)} f(r\cos\theta, r\sin\theta) r dr d\theta\\]

#### 三重积分

**直角坐标系**
\\[\iiint_{\Omega} f(x, y, z) dV = \int_a^b \int_{y_1(x)}^{y_2(x)} \int_{z_1(x,y)}^{z_2(x,y)} f(x, y, z) dz dy dx\\]

**柱坐标系**
\\[x = r\cos\theta, \quad y = r\sin\theta, \quad z = z\\]
\\[dV = r dr d\theta dz\\]

**球坐标系**
\\[x = \rho\sin\phi\cos\theta, \quad y = \rho\sin\phi\sin\theta, \quad z = \rho\cos\phi\\]
\\[dV = \rho^2\sin\phi d\rho d\phi d\theta\\]

### 积分在建模中的应用

#### 概率与统计

**概率密度函数**

连续随机变量X的概率：
\\[P(a \leq X \leq b) = \int_a^b f(x) dx\\]

**期望值**
\\[E[X] = \int_{-\infty}^{+\infty} x f(x) dx\\]

**方差**
\\[\text{Var}(X) = \int_{-\infty}^{+\infty} (x - \mu)^2 f(x) dx\\]

#### 物理应用

**质心计算**

平面薄片的质心坐标：
\\[\bar{x} = \frac{\iint_D x \rho(x, y) dA}{\iint_D \rho(x, y) dA}\\]
\\[\bar{y} = \frac{\iint_D y \rho(x, y) dA}{\iint_D \rho(x, y) dA}\\]

**转动惯量**

绕z轴的转动惯量：
\\[I_z = \iint_D (x^2 + y^2) \rho(x, y) dA\\]

**功的计算**

变力做功：
\\[W = \int_a^b F(x) dx\\]

#### 经济应用

**消费者剩余**

\\[CS = \int_0^{Q^*} [D(q) - P^*] dq\\]

其中 \\(D(q)\\) 是需求函数，\\(P^*\\) 是均衡价格，\\(Q^*\\) 是均衡数量。

**生产者剩余**

\\[PS = \int_0^{Q^*} [P^* - S(q)] dq\\]

其中 \\(S(q)\\) 是供给函数。

## 微分方程：变化的规律

### 常微分方程基础

#### 基本概念

**微分方程**：含有未知函数及其导数的方程
- **阶数**：方程中导数的最高阶数
- **线性**：未知函数及其导数的次数都是1
- **齐次**：不含独立的自由项

#### 一阶微分方程

**可分离变量型**
\\[\frac{dy}{dx} = f(x)g(y)\\]

解法：分离变量后积分
\\[\frac{dy}{g(y)} = f(x)dx\\]
\\[\int \frac{dy}{g(y)} = \int f(x)dx\\]

**一阶线性微分方程**
\\[\frac{dy}{dx} + P(x)y = Q(x)\\]

通解公式：
\\[y = e^{-\int P(x)dx}\left[C + \int Q(x)e^{\int P(x)dx}dx\right]\\]

**贝努利方程**
\\[\frac{dy}{dx} + P(x)y = Q(x)y^n\\]

通过换元 \\(v = y^{1-n}\\) 转化为线性方程。

#### 二阶微分方程

**二阶线性齐次方程**
\\[y'' + py' + qy = 0\\]

特征方程：\\(r^2 + pr + q = 0\\)

根据判别式 \\(\Delta = p^2 - 4q\\) 的符号：
- \\(\Delta > 0\\)：\\(y = C_1e^{r_1x} + C_2e^{r_2x}\\)
- \\(\Delta = 0\\)：\\(y = (C_1 + C_2x)e^{rx}\\)
- \\(\Delta < 0\\)：\\(y = e^{\alpha x}(C_1\cos\beta x + C_2\sin\beta x)\\)

**二阶线性非齐次方程**
\\[y'' + py' + qy = f(x)\\]

通解 = 齐次方程通解 + 非齐次方程特解

### 微分方程组

#### 线性微分方程组

\\[\frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x}\\]

其中 \\(\mathbf{x} = [x_1, x_2, \ldots, x_n]^T\\)，\\(\mathbf{A}\\) 是 \\(n \times n\\) 常数矩阵。

**解的结构**

如果 \\(\mathbf{A}\\) 有 \\(n\\) 个线性无关的特征向量，则通解为：
\\[\mathbf{x}(t) = C_1\mathbf{v}_1e^{\lambda_1 t} + C_2\mathbf{v}_2e^{\lambda_2 t} + \cdots + C_n\mathbf{v}_ne^{\lambda_n t}\\]

### 偏微分方程初步

#### 基本类型

**抛物型**：热传导方程
\\[\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}\\]

**双曲型**：波动方程
\\[\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}\\]

**椭圆型**：拉普拉斯方程
\\[\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0\\]

#### 分离变量法

假设解的形式为 \\(u(x, t) = X(x)T(t)\\)，代入偏微分方程，分离变量求解。

### 微分方程在建模中的应用

#### 人口动力学模型

**Logistic模型**
\\[\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)\\]

解：
\\[P(t) = \frac{K}{1 + \left(\frac{K}{P_0} - 1\right)e^{-rt}}\\]

**捕食者-猎物模型（Lotka-Volterra）**
\\[\frac{dx}{dt} = ax - bxy\\]
\\[\frac{dy}{dt} = -cy + dxy\\]

其中 \\(x\\) 是猎物数量，\\(y\\) 是捕食者数量。

#### 传染病模型

**SIR模型**
\\[\frac{dS}{dt} = -\beta \frac{SI}{N}\\]
\\[\frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I\\]
\\[\frac{dR}{dt} = \gamma I\\]

基本再生数：\\(R_0 = \frac{\beta}{\gamma}\\)

#### 经济增长模型

**索洛增长模型**
\\[\frac{dk}{dt} = sf(k) - (n + \delta)k\\]

其中：
- \\(k\\)：人均资本
- \\(s\\)：储蓄率
- \\(f(k)\\)：人均生产函数
- \\(n\\)：人口增长率
- \\(\delta\\)：折旧率

#### 物理系统建模

**单摆方程**
\\[\frac{d^2\theta}{dt^2} + \frac{g}{l}\sin\theta = 0\\]

小角度近似：
\\[\frac{d^2\theta}{dt^2} + \frac{g}{l}\theta = 0\\]

解：\\(\theta(t) = A\cos(\omega t + \phi)\\)，其中 \\(\omega = \sqrt{\frac{g}{l}}\\)

**阻尼振动**
\\[m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0\\]

根据阻尼系数的大小，有：
- 欠阻尼：振荡衰减
- 临界阻尼：最快回到平衡位置
- 过阻尼：缓慢回到平衡位置

## 级数理论

### 数项级数

#### 级数的收敛性

**正项级数判别法**

1. **比较判别法**：设 \\(0 \leq a_n \leq b_n\\)
   - 若 \\(\sum b_n\\) 收敛，则 \\(\sum a_n\\) 收敛
   - 若 \\(\sum a_n\\) 发散，则 \\(\sum b_n\\) 发散

2. **比值判别法**：\\(\lim_{n \to \infty} \frac{a_{n+1}}{a_n} = \rho\\)
   - \\(\rho < 1\\)：收敛
   - \\(\rho > 1\\)：发散
   - \\(\rho = 1\\)：无法判断

3. **根值判别法**：\\(\lim_{n \to \infty} \sqrt[n]{a_n} = \rho\\)
   - \\(\rho < 1\\)：收敛
   - \\(\rho > 1\\)：发散
   - \\(\rho = 1\\)：无法判断

**交错级数的莱布尼茨判别法**

对于 \\(\sum_{n=1}^{\infty} (-1)^{n-1} a_n\\)，若：
1. \\(a_n > 0\\)
2. \\(a_n\\) 单调递减
3. \\(\lim_{n \to \infty} a_n = 0\\)

则级数收敛。

### 函数项级数

#### 幂级数

\\[\sum_{n=0}^{\infty} a_n (x - x_0)^n\\]

**收敛半径**

\\[R = \frac{1}{\lim_{n \to \infty} \sqrt[n]{|a_n|}} \quad \text{或} \quad R = \lim_{n \to \infty} \left|\frac{a_n}{a_{n+1}}\right|\\]

**阿贝尔定理**：幂级数在收敛圆内绝对收敛，在收敛圆外发散。

#### 傅里叶级数

**三角级数形式**

\\[f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx)\\]

**傅里叶系数**

\\[a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos nx \, dx \quad (n = 0, 1, 2, \ldots)\\]

\\[b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin nx \, dx \quad (n = 1, 2, 3, \ldots)\\]

**复指数形式**

\\[f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}\\]

其中：\\[c_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x) e^{-inx} dx\\]

### 级数在建模中的应用

#### 函数逼近

用有限项级数近似复杂函数，如：
- 计算器中的三角函数和指数函数
- 数值分析中的插值和拟合

#### 信号处理

傅里叶级数在信号分析中的应用：
- 频谱分析
- 滤波器设计
- 图像压缩

#### 数学物理

在求解微分方程时，常用级数展开：
- 边界值问题的本征函数展开
- 格林函数的级数表示

## 变分法基础

### 变分问题的提出

**最短路径问题**：在所有连接两点的曲线中，找到使某个积分取极值的曲线。

**一般变分问题**：在函数类中找到使泛函
\\[J[y] = \int_a^b F(x, y, y') dx\\]
取极值的函数 \\(y(x)\\)。

### 欧拉-拉格朗日方程

如果 \\(y(x)\\) 使泛函 \\(J[y]\\) 取极值，则必须满足：

\\[\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0\\]

这就是著名的欧拉-拉格朗日方程。

### 变分法的应用

#### 最速降线问题

质点在重力作用下沿曲线滑动，求使时间最短的曲线形状。

泛函：\\[T = \int_0^a \frac{\sqrt{1 + (y')^2}}{\sqrt{2gy}} dx\\]

解得最速降线是摆线。

#### 等周问题

在所有周长相等的闭合曲线中，求围成面积最大的曲线。

答案：圆形。

#### 物理中的应用

**费马原理**：光沿光程最短的路径传播
**最小作用量原理**：物理系统沿使作用量最小的路径演化

## 小结

微积分是数学建模的核心工具，它提供了：

1. **描述变化的语言**：导数描述瞬时变化率，积分描述累积效应
2. **优化的方法**：通过求导找极值，解决最优化问题
3. **建立模型的框架**：微分方程描述动态系统的演化规律
4. **分析工具**：级数展开、变分法等高级技巧

掌握微积分不仅要理解其计算技巧，更要理解其几何意义和物理意义，能够将实际问题转化为数学问题，运用微积分工具求解，并能正确解释结果的实际意义。

在后续学习中，我们将看到微积分如何与线性代数、概率统计等其他数学工具结合，构成强大的数学建模工具箱。
