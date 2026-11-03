# v4：最小充分标量矩表示与共同认证

## 0. 结论的边界

这是经典 Newton/Prony/矩反演的可逆编码，不主张新方法、神经元实现或原 TAN 的性能改变。所谓“最小”指本轮所选充分矩通道集合为 2L 个实标量，不主张允许不连续/无限精度编码时的一般信息论最小性。它没有消除键的信息：理想状态可解码出完整无序 key/value 表。

有限实现的安全性依赖：(i) 真实输入误差有有效上界；(ii) 所有不同 canonical 身份（包括未来写入及需要做 OLD 认证的查询身份）遵守外部最小间隔 Delta；(iii) 持久状态从空状态经认证事务演化，无预算外保持漂移、计数损坏或位翻转；(iv) 区间认证运算正确。Delta 不是算法从当前矩阵自行推断出的未来承诺。本轮只模拟定点写入舍入，不模拟未知硬件漂移。

## 1. 精确表示：variable N、空状态、membership 与 latest

容量 L>=1；当前 N<=L 个互异实键 a_i 属于 [-1,1]，latest 值为 t_i。

持久 payload：

m_j = sum_i a_i^j, j=1,...,L;
b_j = sum_i t_i a_i^j, j=0,...,L-1.

另存精确整数 N，m0=N 不作为实标量另存。

Newton 恒等式：e0=1，
r e_r = sum_{j=1}^r (-1)^(j-1) e_{r-j} m_j，r=1,...,N。

所以前 N 个幂和确定
p_N(z)=z^N-e1*z^(N-1)+...+(-1)^N e_N=prod_i(z-a_i)。

它的 N 个简单实根唯一确定无序键集合。对恢复出的根，定义 A_{ji}=a_i^j，j=0,...,N-1。Vandermonde 行列式为两两键差乘积，非零，故前 N 个值矩 b0,...,b(N-1) 唯一确定 latest 值。

更直接地，Lagrange 多项式
ell_i(z)=prod_{h!=i}(z-a_h)/(a_i-a_h)=sum_{j=0}^{N-1} ell_ij z^j
满足 t_i=sum_j ell_ij b_j。这是实际值区间证书所用公式。

### 空状态与计数不可省略

N=0 时 p_0(z)=1，没有任何旧键，所有 payload 初始为零。空集合与 {(0,0)} 的所有已存 m1..mL、b0..b(L-1) 都是零，但 N 分别为0和1。因此 count 对此表示是必要控制信息，不能由 payload 自行推断。

### 精确 membership 不依赖 spare mode

q 是已存键当且仅当 p_N(q)=0。N=L 时 p_L 仍可测试未知查询。p_L(q)!=0 的写入必须先判 NEW，再按容量拒绝；p_L(q)=0 的重复键仍可覆盖。此时 degree-(L-1) 特征 span 可以已满，但该 span 判据根本不是本架构的 membership 规则。

精确查询甚至可不恢复所有根：已知 p_N(q)=0 时，ell_q(z)=p_N(z)/((z-q)p_N'(q))，直接与前 N 个值矩作内积。有限噪声版本本轮没有实现这一路；它仍暂存并认证全部根区间，不能把这个解析观察当成运行时节约。

### 精确更新不变量

NEW a：N<L 时，对全部存储阶次执行 m_j+=a^j、b_j+=v a^j，N+=1。
OLD a_i：先恢复 t_i，令 r=v-t_i；m 完全不变，b_j+=r a_i^j，N 不变。

逐项代入可知 NEW/OLD 均保持 m_j=sum_distinct a_i^j 与 b_j=sum_latest t_i a_i^j，因此恢复其他键的值不变，被写键变为 v。所有存储阶次即使暂时未用于解码，也必须更新；不能只更新当前 N 个通道，否则未来 N 增大后不变量不成立。

同键反复写不累加 key moments，不增加 N。未知查询返回 absent，而不是把在非根处算出的插值值冒充 membership。

## 2. 噪声语义：canonical 身份与 first-anchor 分开

真实键是 K_i，观测 x 满足 |x-K_i|<=zeta。首次接纳时登记 a_i=round_h(x)，之后 canonical 身份的 latest 值围绕固定 a_i 编码。OLD 写不把 anchor 改成当前 noisy x。

持久 sigma 保证 |a_i-K_i|<=sigma。NEW 时更新
sigma'=max(sigma, ceil_h(zeta+|a_i-x|))。

恢复出的登记根区间为 I_i；canonical 区间 J_i=(I_i+[-sigma,sigma])交[-1,1]；输入区间 X=[x-zeta,x+zeta]交[-1,1]。

OLD 充分条件：X只与一个J_i相交，且 max_{u in X,v in J_i}|u-v|<Delta。
NEW 充分条件：对所有i，dist(X,J_i)>=Delta。
否则拒写/拒答。空集合下 NEW 条件自然成立。

OLD 正确性：如果真实输入属于另一个合法身份，则它与 K_i 的距离必须>=Delta，而两者分别落在 X、J_i，违背 max-distance<Delta。NEW 正确性：真实旧键一定同时落在 X 和其J_i，因此不能与所有J_i隔开。

无未来间隔承诺时，这个OLD结论不成立；一个很近的新键可与重复键产生同一观测。区间正好跨多个旧地址同样必须拒绝或额外提供身份信息。

## 3. 删除 Hankel 与高阶一致性门之后，用什么保证安全？

维护来源性不变量：
max_{stored j}|mtilde_j-sum a_i^j|<=mu；
max_{stored j}|btilde_j-sum t_i a_i^j|<=beta。

v4对冗余和最小表示都只用 N、m1..mN、b0..b(N-1)、mu、beta、sigma 认证。它不计算或要求 Hankel 谱门，也不读取更高阶矩作一致性判定。

### 根证书

把每个 m_j 放入 [mtilde_j-mu,mtilde_j+mu]，精确有理区间 Newton 递推包含真正 p_N 的系数。numpy.roots 只提出中心。冻结程序尝试半径 2h*2^r，r=0,...,40；要求 N 个区间在 [-1,1] 内且互不相交，并使每个区间的两个端点对所有允许系数严格异号。

真正多项式每个区间至少有一个根；次数为N而已有N个互不相交区间，所以它们穷尽全部根且每区间唯一。靠近边界、病态或提议失败时可能拒绝；不把拒绝称为不可能恢复。

### 值证书

用根区间构造 interval Lagrange 系数 L_ij，令 T_i=sum_j L_ij*[btilde_j-beta,btilde_j+beta]。
中心根和存储值矩给出点估计 that_i。
E_i=max(|lower(T_i)-that_i|,|upper(T_i)-that_i|) 包含真实 committed t_i 的误差。

若候选的全部 E_i<=epsilon，结合根/地址证明就足以保证被提交事务所有已承诺键的读取误差。无需再要求与本直接输出证书不同的 Hankel 条件数门。

这不等于增加了无故障假设之外的容错：删除冗余检查减少了可用的完整性诊断。若 mu/beta 的来源不可信，或count/anchor历史损坏，仅找到某些实根不能证明真实记忆还在。

## 4. 非破坏性事务与同一个保守预算

h=2^-p。持久标量最近舍入误差<=h/2，预算向上舍入到h的格点。
NEW：mu'=ceil_h(mu+h/2)，beta'=ceil_h(beta+h/2)。
OLD：r=v-that_i，中心与真anchor距离<=r_i。

逐个坐标 j，
|btilde'_j - b'_j| <= beta + E_i + |r| j r_i + h/2，
因为 |a_i|,|ahat_i|<=1，|ahat_i^j-a_i^j|<=j r_i。

为隔离payload差异，两种矩表示在本轮都使用
beta'=ceil_h(beta+E_i+|r|Lr_i+h/2)。

候选快照中的计数、mu/beta/sigma与全部payload验证完后一起提交；否则原snapshot字节完全不变，不投影b，不修补矩，不删除弱记忆。拒绝的请求另列为未完成目标。

### 可合法缩紧、但本轮绝未启用的界

最小表示最高值矩阶次为 L-1，所以可以合法改成
beta'_minimal<=ceil_h(beta+E_i+|r|(L-1)r_i+h/2)。

这一改变属于误差证书改进，不是单纯少存几个数带来的性能。它未用于本轮任何结果，也不能在只有N个当前键时再擅自用N-1替代L-1：尚未使用的高阶存储值矩仍须对未来准入有效。

## 5. 共同证书的等价性预言与解释

初始时两种矩表示的 m1..mL、b0..b(L-1)、N、mu、beta、sigma相同。解码和决策只访问这些共享前缀。

归纳地：同请求得到同roots/values/residual/candidate预算和决策；新增逐坐标舍入保持共享前缀相同；OLD也是如此；拒绝不改任何一方状态。因此整条请求流中两者的接受率与读取值应逐项相同。多余高阶矩会继续更新，但不能影响共同证书。

这是一项原则性表示比较，不是期望minimal自动提高coverage的竞赛。与旧v3门控不同的收益如有发生，必须归因于共同证书变化；实际本轮没有coverage收益。

## 6. 有限精度解释不能越界

三个条件不构成总体成功率实验；无随机样本、bootstrap、训练或调参。
表使用同一noisy observation、sigma与Delta合同，没有真key oracle。目标值恰是低阶二进制有理数，所以表本轮为零误差；不推广到任意实值。

精确代数对照只检查变量N、零键、重复、满容量及unknown membership，不是同bit硬件。即使core同为2L，最小矩还多两个动态预算word，且求根与区间工作空间/时延非零。

不改变原TAN/P1-B/v11，也不证明生物可行性、能耗优势、无间隔连续地址认证或无限次覆盖coverage。
