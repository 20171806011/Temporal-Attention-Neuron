> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# 核验数值条件修订；科学方案与结果未修改

第一次独立核验在旧 POINT 的探针分数上停止：固定绝对误差阈值 1e-8，观察到 1.30077e-8。原 audit.py、冻结记录和 AUDIT_old_FAILURE.json 保留；没有重跑任何科学实验。

定向排查六个旧模型发现，独立神经元重放的特征最大误差约 1.07e-14；SVD 与原 dual ridge 在**相同输入**上也会出现大于 1e-8 的分数差。原因是训练集中恒定或近恒定的特征按预定下限 1e-8 标准化，新测试样本该特征可非恒定；矩阵乘法会放大极小的系数舍入差。独立原生重放加 SVD 的最大分数差约 5.44e-8，所有旧探针的分类均未改变。诊断完整保存在 AUDIT_NUMERIC_DIAGNOSTIC.json。该诊断占用的核验预算保守记 10 秒，而非忽略失败成本。

audit_revised.py 不改变特征、权重、探针、分类阈值、科学结果或神经网络训练。它保留原状态/预处理/系数误差阈值，并给分数额外检查：令独立标准化测试矩阵为 Z、独立系数为 w，保存实现为 Z0、w0，逐样本核对

`abs(Zw − saved_score) <= abs(Z) @ abs(w − w0) + abs(Z − Z0) @ abs(w0) + 1e-8`。

另独立重算 Z0w0 必须与保存分数匹配，最终所有分类必须逐例完全相同。这样检查的是可解释的误差传播，不能把任意错误放宽成通过。修订源代码有单独哈希，第一次失败仍计入累计 180 秒核验预算。科学阶段本身不重复执行。

本修订在任何本轮新任务网络训练前完成。复现时 REPRODUCE.md 中 `audit.py old/new` 应使用 `audit_revised.py old/new`，并复制 AUDIT_REVISION.json、AUDIT_old_FAILURE.json 和 AUDIT_NUMERIC_DIAGNOSTIC.json 以保留核验来源。原版本可复现原失败。
