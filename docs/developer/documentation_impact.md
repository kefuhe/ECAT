# 代码—文档影响评估

本文说明代码修改完成前怎样判断、更新和验证 ECAT 公开文档。目标不是让每个代码 diff
都产生文字改动，而是保证用户实际看到的接口、配置、科学约定、输出和工作流与当前实现一致。

## 1. 先判断用户可见行为

```text
代码 diff
  -> 找到可能受影响的 workflow、example、reference 和测试
  -> 对照真实签名、parser、模板、输出和科学约定
  -> 只更新发生用户可见变化的页面
  -> 构建文档并运行聚焦测试
  -> 在任务结果中报告 Docs impact
```

机器路由只回答“哪些位置需要重新检查”，不能替代接口和科学判断。候选页面最终没有修改时，
维护者仍应说明现有内容为什么继续准确。

## 2. 哪些变化需要更新公开文档

| 改动性质 | 公开文档处理 |
| --- | --- |
| 公开 API、CLI、YAML 字段或默认值变化 | 更新对应 workflow/example/reference，并核对最小示例 |
| 输出文件、表格字段或用户可见错误变化 | 更新输出和故障排查说明 |
| 公式、单位、坐标、符号或参数排列变化 | 更新概念或 reference，并补相应数值验证 |
| 修复实现以恢复已有公开契约 | 现有文档正确时可不改；必要时补充用户可操作的诊断 |
| 内部重构，接口、结果和工作流完全不变 | 通常不改公开文档 |
| 纯性能优化，数值结果和用户操作不变 | 通常不改；若新增公开设置或资源建议则更新 reference |
| 一个研究项目的目录或运行习惯 | 不进入通用手册；只有稳定公开案例才进入 casebook |

公开页面只说明用户需要的稳定事实。源码内部对象所有权、缓存实现、临时变量和调试过程不应
因为一次代码修改而被复制进 workflow 或 reference。

## 3. 标准执行顺序

### 3.1 修改前

1. 阅读目标模块、现有测试和当前公开规范入口。
2. 明确行为由 eqtools、CSI 还是两者的公开交接接口负责。
3. 在 ECAT-Cases 中检查稳定公开用法；案例是兼容性证据，不覆盖当前源码、明确契约和科学约定。
4. 如果公开案例本身需要修改，先单独说明范围、原因、用户影响和验证方法。

### 3.2 修改后

从 eqtools 仓库运行：

```bash
python maintainer_tools/check_docs_impact.py --repo eqtools
```

分析显式路径或 CSI diff 时可以使用：

```bash
python maintainer_tools/check_docs_impact.py \
  --repo csi --paths csi/TriangularPatches.py csi/edge_utils/topology_boundary.py
```

重点读取以下结果：

- `public_docs`：必须重新核对的公开页面；
- `tests`：建议运行的聚焦测试；
- `unmapped_code_paths`：尚未覆盖的代码路径，需要人工判断是否补充映射。

`public_docs` 通常指向单个 Markdown 页面，也可以用 `/**` 指向需要整体复核的页面集合。
映射校验要求单页真实存在、集合至少匹配一个文件；路由结果保留集合写法，不把维护中的
目录展开成一份容易随页面增删而过期的重复清单。

路由表可能包含项目维护使用的其他组织字段。它们不定义公开 API，也不应被复制到用户手册。

### 3.3 完成前

至少检查：

1. 文档中的方法、参数和默认值与真实签名一致；
2. YAML 示例与配置解析器、生成模板一致；
3. CLI 示例与 parser 一致；
4. Python/YAML 代码块可以解析；
5. 公开文档不包含本机绝对路径或无法公开访问的材料；
6. 相对链接和 ECAT-Cases 链接可以解析；
7. 聚焦测试通过，跨模块行为按风险扩大测试范围；
8. `git diff --check` 没有空白错误。

代码验证不能只确认“能够运行”。公式、单位、坐标、正负号、参数布局、矩阵和 patch 索引、
缓存有效性以及 MPI 结果都是科研正确性约束。任何无法解释的数值漂移都应阻断完成。

## 4. 文档层级怎样选择

| 用户问题 | 主要落点 |
| --- | --- |
| 第一次怎样安装和跑通 | `getting_started/` |
| 怎样完成一项科研任务 | `workflows/` |
| 为什么采用这种对象或科学约定 | `concepts/` |
| 最短可复制代码怎样写 | `examples/` |
| 哪个真实公开案例可以对照 | `casebook/` |
| 字段、公式、API 和边界条件是什么 | `reference/` |

一次功能通常只需要一篇 workflow、一个短例和 reference 中一个规范小节。不要把完整参数表
复制到 workflow，也不要把实现审计写成用户教程。

## 5. 跨 eqtools 与 CSI 的公开说明

CSI 负责 mesh、source、Green's function 和 Laplacian 等底层能力，eqtools 负责配置、adapter、
pipeline 和反演调度。底层变化如果影响用户可见接口或结果，必须同时检查 eqtools 的上层页面；
如果公开行为没有变化，则不在用户手册中复述跨包内部调用链。

## 6. 结果报告格式

发生公开文档变化时可写：

```text
Docs impact: public updated
Updated: geometry perturbation reference
Verified against: method signature, generated template and focused tests
```

不需要修改公开文档时可写：

```text
Docs impact: none
Reason: internal variable rename; API, numerical result and documented workflow are unchanged.
```

机器映射位于 `maintainer_tools/docs_impact_map.yml`。它是候选路由表，不是功能清单；新增规则
应使用稳定的仓库相对路径，并只指向实际存在的页面、至少包含一个页面的集合和测试。
