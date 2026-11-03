# v3 数学与认证边界

本文件的定理是条件数学推导；有限集合结果另见 results_v3.json。这里不是原 TAN/P1-B/v11 的实验或修订。Prony、Newton 恒等式、Vandermonde 插值与矩反演是既有方法，不主张新颖性。

## 1. 精确模型及 intrinsic membership

令 L=Nmax, d=L+1, f(k)=(1,k,...,k^L)^T。已存 N<=L 个互异标量键 k_i。设 F=[f(k_i)]。

任意新键 k 不等于所有 k_i，增广矩阵 [F f(k)] 的前 N+1 行是方形 Vandermonde，其行列式为所有两键差的乘积且非零。因此 f(k) 不在 range(F)。这在 N=L 时仍成立：必须先识别新键，然后因管理容量 Nmax 拒绝，而不是把它判为旧键。系统保留至少一个空模式，不能准许最终写到 d 个不同键后再沿用这个 membership 保证。

持久统计量：

m_j=sum_i k_i^j, j=0,...,2L;
b_j=sum_i t_i k_i^j, j=0,...,L;
C_ab=m_{a+b}, a,b=0,...,L.

m_0=N；C是Hankel矩阵，不需要独立存储 d(d+1)/2 个矩阵元素。

令 w=C^dagger b。列独立给出 F^T C^dagger F=I，因此 f(k_i)^T w=t_i。

新键：m_j += k^j，b_j += v k^j。旧键：m完全不动，r=v-f(k_i)^T w，b_j += r k_i^j。由 b=Ft 与上述dual恒等式，新旧更新都保持latest不变量，旧更新不修改其他关联。空状态全零。不重复累计旧键的 key moments 是必要条件。

## 2. 重建键不是免费的“隐形”操作

定义 p(z)=prod_i(z-k_i)=z^N-e_1z^(N-1)+...+(-1)^N e_N。
Newton恒等式：e_0=1，r e_r=sum_{j=1}^r (-1)^(j-1)e_{r-j}m_j。

因为所有键的统计权重均为1，已知N时，m_1,...,m_N 就确定 p，进而确定无序键集合。另一种常见表述是 annihilation：若 p(z)=sum_{r=0}^N a_r z^r，则 sum_r a_r m_{j+r}=0。这就是经典有限支撑矩/Prony反演结构。

所以本编码是键表的可逆坐标表示，不是删除了键身份信息。本实现每次操作瞬时重建根，没有持久化根表；但为求根、区间和插值系数支付工作空间与时延。

membership 在精确无噪声条件下等价于 p(q)=0。query 可在已认证根处求 f(q)^T C^dagger b。实现采用等价的 Lagrange 值恢复：

ell_i(z)=prod_{h!=i}(z-k_h)/(k_i-k_h)=sum_{j=0}^{N-1} Ainv_ij z^j，
t_i=sum_j Ainv_ij b_j。

有限精度时，本实现不是对带噪q直接计算朴素伪逆；它先认证地址，再输出对应登记节点恢复出的值。这一额外地址处理必须计账。

## 3. 有噪新登记的语义

canonical真实键为K_i，传感器观测x满足|x-K_i|<=zeta。新登记内部锚点a_i=quantize(x)。之后latest值仍对应canonical身份K_i，而统计量围绕固定a_i编码；不会在每次重复观测时把锚点漂移到新x。

持久sigma保证每个|a_i-K_i|<=sigma。它在新登记时更新为max(sigma,zeta+|a_i-x|)的上舍入值。后续旧写不增加锚点误差。

外部输入合同：所有不同canonical键（包括将来到达的新键）至少相隔Delta。Delta不是从当前矩阵臆测出来的；sensor bound也不是从评分器真键算出后喂给学习器。初始状态为空且误差为零。当前试验只模拟确定的定点写入误差，不模拟未知自主保持漂移。

如果允许任意近的连续新键，在非零输入噪声下无法统一判定重复vs新键：已存0、观测0，可以来自旧0，也可以来自足够近的新h及噪声-h。无论Vandermonde在精确算术下多好，这种观测歧义仍然存在。

## 4. 根区间与可检查地址证书

状态证书为 ||mtilde-m(a)||_infinity<=mu。将每个m_j替成[mtilde_j-mu,mtilde_j+mu]，用精确有理数区间Newton恒等式得到所有可能的真实多项式系数范围。

numpy.roots只提出中心。随后在预定半径倍增程序中寻找N个互不相交实区间，使区间多项式在每个左/右端点对所有允许系数都严格异号。由介值定理和多项式次数N，每个区间恰有一个根，且已找到全部根。再用更高阶矩作区间一致性检查。找不到证书就拒绝；没有把root proposal当证明。

令根区间为I_i，canonical键区间J_i=I_i+[-sigma,sigma]（与[-1,1]相交），输入区间X=[x-zeta,x+zeta]与[-1,1]相交。

OLD充分条件：X只与一个J_i相交，且max_{u in X,v in J_i}|u-v|<Delta。
NEW充分条件：对每个i，dist(X,J_i)>=Delta。
其他情况全部拒绝。旧判据排除了同一观测来自一个合法新键的可能，而不仅仅是“接近旧根”。新判据在满容量时仍能认证new，然后容量拒绝。

这些是保守充分条件；拒绝不证明该具体观测一定无法恢复。

## 5. 内容及谱证书

维护 beta，使 ||btilde - sum_i t_i f(a_i)||_infinity<=beta，t_i为已commit的latest值。

利用根区间构造区间Lagrange系数L_ij；令真实值区间T_i=sum_j L_ij [btilde_j-beta,btilde_j+beta]。
中心根和存储b给出点估计that_i；E_i=max(|lower(T_i)-that_i|,|upper(T_i)-that_i|)严格覆盖其误差。

以A_ji=a_i^j（j=0,...,N-1）定义方形Vandermonde。Lagrange系数是A^{-1}的行。区间绝对值给出其1范数上界B1和无穷范数上界Binf，故

alpha_lower=1/(B1 Binf) <= sigma_min(A)^2 <= lambda_min(F^T F)。

当每个Hankel条目的误差<=mu时，||Ctilde-C||_2<=(L+1)mu=deltaC。
冻结的额外谱安全门为alpha_lower>4deltaC。这里不靠特征值截断修补C，不投影b。

根逼近的难度不能由满秩消除：p'(a_i)=prod_{j!=i}(a_i-a_j)，近键会放大系数扰动。另有直接的span下界：p的系数向量h与所有旧特征正交，因此dist(f(k),span(F))>=|p(k)|/||h||。若新键距每个旧键至少Delta_new，则保守地>=Delta_new^N/2^N。阶数或近键都会使可用margin很小。

## 6. 候选事务与误差传播

内部存储步长h=2^{-p}，逐标量最近舍入误差<=h/2；误差预算自身向上舍入到h的格点。临时计算及区间认证使用精确有理数，不能称为同64位物理硬件。

NEW：以新锚点a精确计算幂后更新m,b并舍入。mu^+<=mu+h/2，beta^+<=beta+h/2，再将两个预算向上舍入。

OLD：已认证根i，中心ahat_i的根误差半径为r_i，关联残差r=v-that_i。只对候选b做

btilde^+=quantize(btilde+r f(ahat_i))。

m完全不动。因为|a_i|,|ahat_i|<=1，逐坐标特征误差<=Lr_i，且真实old value误差<=E_i，所以

beta^+<=beta+E_i+|r|Lr_i+h/2，

随后向上舍入。该界不利用误差抵消，非常保守；连续写入可能使证书快速耗尽。

候选必须同时通过地址、容量、根、谱门和所有E_i<=epsilon才commit。否则不写任何持久字段，包括N、mu、beta、sigma。候选状态可以由外部审计日志保留，但不会送回学习器。

正确的b可能位于真实range(C)，有限精度b也可能有小的range外误差。本方案不删除它，而是把它纳入beta并在未来插入后重新认证；它没有声称消灭污染，只承诺被接受的事务仍满足完整内容误差界。

## 7. 读出、拒绝、比较与限制

读查询与写查询使用相同的有界输入噪声和地址证书。通过则返回that_i；新键可以认证为不存在；其他情况拒答。地址拒绝、容量拒绝和内容证书失败都不计为完成了写入。

显式表存同一首次带噪锚点和定点value，使用相同的sigma/Delta地址认证，绝不获得无噪声canonical key。理想矩对照使用同样的带噪观测，只把内部算术改为精确有理数；它同样拒绝地址歧义和满容量新键。

本试验所有目标值是二进制有理数，显式表在所选精度恰能精确表示；其零误差不应外推为任意实值无量化误差。

Chebyshev分析（未实验）：T_iT_j=(T_{i+j}+T_{|i-j|})/2，因此也能用2L+1个Chebyshev矩组织C。它是可逆基变换，保持精确membership；可能改变数值条件，但不能消除近键本身的观测歧义，也不能不变换噪声/尺度就比较raw condition。本轮没有挑选更好基或调参。

## 8. 能承诺与不能承诺

已证明：精确有限键集合的intrinsic membership及latest更新；在所述来源误差、最小间隔合同和精确区间算术下，接受事务的值误差及拒绝不变性。
未证明：未知硬件漂移的处理、无限次覆盖始终有coverage、无最小novelty间隔的membership、物理能耗优势、细胞实现、总体统计成功率，或任何原TAN/v11结论的更改。
