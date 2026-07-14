# Mooncake 项目知识梳理

> 本文件由逐步梳理累积而成,每次只补充"必要的部分"。
> 最近更新:2026-07-14 · 已梳理:① 顶层架构与模块地图 ② Transfer Engine 架构与调用流程 ③ TE 旧实现:RDMA 层(含深入细节) ④ TENT 新实现(含深入细节) ⑤ Mooncake Store 架构 ⑥ Mooncake EP / PG

---

## ① 顶层架构与模块地图

### 项目定位
Mooncake 是面向大规模 **LLM 推理与训练** 的基础设施,核心是 **以 KVCache 为中心的解耦(disaggregated)架构**:
- 将 **prefill(预填充)** 与 **decode(解码)** 集群分离;
- 复用 GPU 集群中闲置的 **CPU / DRAM / SSD** 资源,构建解耦的 **KV cache 池**。
- 是 Moonshot AI 旗下 **Kimi** 的服务平台;在真实负载下让 Kimi 在满足 SLO 的前提下多处理 75% 的请求。
- 论文:FAST'25(`FAST25-release/` 含 traces);已加入 PyTorch 生态。

### 分层结构(自底向上)
```
   Tensor 为核心数据载体,贯穿全栈
   ┌─────────────────────────────────────────────┐
   │ 生态集成:SGLang / vLLM / TensorRT-LLM / LMCache│
   ├─────────────────────────────────────────────┤
   │ Mooncake EP + PG   → 弹性容错 MoE 分布式执行     │
   │ Mooncake Store     → 分布式 KV cache / 权重管理  │
   │ Transfer Engine(TE)→ 跨异构网络/加速器的数据搬运 │
   └─────────────────────────────────────────────┘
```

### 核心模块(顶层目录)
| 目录 | 角色 | 说明 |
| --- | --- | --- |
| `mooncake-transfer-engine/` | **TE,项目核心** | 统一的高性能批量数据传输框架。多协议(TCP/RDMA/EFA/NVMe-oF/NVLink/HIP/CXL/Ascend 等)、拓扑感知路由、多网卡带宽聚合、自动故障转移。含 `tent/`、`rust/`、多种 allocator。 |
| `mooncake-store/` | **分布式 KV 缓存存储引擎** | 基于 TE 构建。对象存储、复制、淘汰(eviction)、软/硬 pin、多级缓存(DRAM+SSD/NVMe)、大对象 striping、零拷贝。含 master service、`go/`、`rust/`。 |
| `mooncake-ep/` | **Expert Parallel(MoE)** | DeepEP 风格的 expert-parallel dispatch/combine,带 `active_ranks` 容错感知。Python 扩展(`setup.py`)。 |
| `mooncake-pg/` | **Process Group** | PyTorch `torch.distributed` 后端;支持 all_gather 等集合通信,可检测失败 rank、上报、无需重启恢复 rank。 |
| `mooncake-p2p-store/` | P2P store | 点对点存储(独立小模块,含 `build.sh`)。 |
| `mooncake-rl/` | RL 示例 | 仅 `examples/`,强化学习集成示例。 |
| `mooncake-integration/` | 生态/Python 集成 | store 与 transfer_engine 的 Python 绑定、allocator(含 Ascend NPU)。 |
| `mooncake-common/` | 公共库 | etcd、k8s-lease、CMake 查找模块(GLOG/JsonCpp/Mpi/Urma)、PyTorch/Python 环境配置。 |
| `mooncake-wheel/` | 打包 | Python wheel 构建。 |

### 构建 / 生态
- C++ 为主(CMake,顶层 `CMakeLists.txt`),辅以 Rust、Go、Python 绑定。
- 依赖安装脚本:`dependencies.sh`;本地 CI:`scripts/run_ci_test.sh`(见 skill `mooncake-ci-local`)。
- PyPI 包按硬件分发:`mooncake-transfer-engine`(CUDA≤12.9)、`-cuda13`、`-non-cuda`、`-npu`。

---

## ② Transfer Engine(TE)架构与调用流程

代码位置:`mooncake-transfer-engine/`(include/ + src/)。

### 分层类结构
```
   TransferEngine (transfer_engine.h)          ← 用户 API 门面(Pimpl)
      ├── impl_      : TransferEngineImpl       ← 传统实现
      └── impl_tent_ : tent::TransferEngine     ← 新实现(TENT,use_tent_ 开关,见 tent/)
                │
                ▼
   MultiTransport (multi_transport.h)           ← 多协议调度器
      transport_map_: proto名 → Transport       ← 例:"rdma"/"tcp"/"nvlink"/"nvmeof"/"cxl"/"hip"...
                │  selectTransport() 按目标 segment 的 protocol 选路
                ▼
   Transport (transport/transport.h, 抽象基类)   ← 各协议实现,src/transport/<xxx>_transport/
      submitTransferTask() 把 Request 切成 Slice 下发硬件
```
- **门面用 Pimpl**:`TransferEngine` 持 `shared_ptr<TransferEngineImpl>`,并有 TENT 新实现二选一(`use_tent_`)。
- 具体协议目录:`rdma_transport`(核心)、`tcp_transport`、`nvlink_transport`、`nvmeof_transport`、`cxl_transport`、`hip_transport`、`efa_transport`、`ascend_transport`、`kunpeng_transport`、`barex_transport`、`cxi_transport`、`maca_transport`、`intranode_nvlink_transport`、`sunrise_link`、`rpc_communicator`、`device`(IBGDA/P2P GPU 侧)。

### 核心数据结构(transport.h)
| 结构 | 作用 |
| --- | --- |
| `TransferRequest` | 单次传输请求:opcode(READ/WRITE)、source 本地地址、target_id(SegmentID)、target_offset、length。`transport_hint` 为 TENT 的 per-request 传输固定。|
| `BatchID` | 不透明 uint64,**直接 reinterpret 成 `BatchDesc*`**(热路径省去 map 查找);`toBatchDesc()` 转换。|
| `BatchDesc` | 一个 batch:`task_list`(vector<TransferTask>)、batch_size、完成计数、`is_finished`/`has_failure` 原子标志、完成用 mutex+cv(事件驱动完成)。|
| `TransferTask` | 一个 Request 拆成多个 Slice 后的聚合体,记 slice_count / success / failed / transferred_bytes / is_finished,持 `transport_` 指针用于状态轮询。|
| `Slice` | **最小传输单元**(堆分配,完成即自删)。含各协议专属 union(rdma/ub/local/tcp/nvmeof/cxl/hccl/ascend_direct/ubshmem)。`markSuccess/markFailed` 原子累加并触发 batch 完成检查。|
| `ThreadLocalSliceCache` | 线程本地 Slice 复用池(容量 4096),避免热路径频繁 new/delete。|

### 关键辅助模块
- **TransferMetadata**(transfer_metadata.h + plugin):管理 **Segment 元数据**(每个远端注册的内存区域=Segment,含 buffers、协议、rkey 等),通过 metadata server 交换(插件式,如 etcd/redis/http)。selectTransport 靠它 `getSegmentDescByID` 拿到目标协议。
- **Topology**(topology.h):**拓扑感知设备选择**。`discover()` 探测本机 HCA 列表;`selectDevice(location, hint, retry)` 按 NUMA/位置亲和从 preferred_hca/avail_hca 选最优 RDMA 网卡,失败可换路(retry_count 递增实现 **多网卡故障转移/带宽聚合**)。
- **transfer_engine_c.cpp**:C ABI 封装(供 Python/Rust/Go 绑定)。

### 主要调用流程(以 RDMA WRITE 为例)
```
1. init(metadata_conn, local_server_name, ip, rpc_port)
      → 连接 metadata server、installTransport("rdma") → Topology.discover() 探测网卡
2. registerLocalMemory(addr,len,location)   ← 注册本地内存(RDMA reg_mr 得 lkey/rkey),写入 metadata
3. openSegment("remote_node")               ← 拿到远端 SegmentID(handle)
4. batch = allocateBatchID(n)               ← new BatchDesc,id 即指针
5. submitTransfer(batch, [requests])
      MultiTransport::submitTransfer:
        for each request:
          selectTransport()   ← 按 target segment 的 protocol 选 Transport(多协议段按优先级 hip>cxl>rdma>tcp 选路)
          归组 submit_tasks[transport]
        for each transport: transport->submitTransferTask(tasks)
6. RdmaTransport::submitTransferTask:
        for each task:
          selectDevice()      ← Topology 选本地 HCA(NUMA 亲和 + 失败重试换网卡)
          按 slice_size 切成多个 Slice(尾部 <block+fragment 合并)
          填 dest_addr/rkey/lkey → 按 RdmaContext(每设备)归组
        批量 post 到 QP(达到 watermark 即 flush)
7. getBatchTransferStatus(batch) / getTransferStatus(batch, task_id)
        轮询或事件驱动:每个 Slice 完成 markSuccess→原子累加→最后一个触发
        batch_desc.is_finished + cv.notify → 唤醒等待者
8. freeBatchID(batch)
```

### 设计要点
- **切片(Slice)**:大传输按 `globalConfig().slice_size` 切分,天然支持**多网卡并行 I/O 与带宽聚合**;每个 slice 独立选设备/重试。
- **零拷贝**:source/dest 都是已注册内存(RDMA MR),硬件直接 DMA。
- **故障转移**:`selectDevice` 的 retry_count 递增,在多张 HCA 间自动换路;Request 层 `advise_retry_cnt`。
- **完成通知**:编译开关 `USE_EVENT_DRIVEN_COMPLETION` 走 cv 直接唤醒,否则轮询。
- **多协议段**(`ENABLE_MULTI_PROTOCOL`):一个 Segment 可注册多协议(如 "rdma,hip"),单 batch 内按 buffer 覆盖的地址 + 优先级逐请求分流(intra-node 走 hip GPU-IPC,cross-node 自动回落 rdma)。
- **TENT**:新一代实现(`tent/`),支持 per-request `transport_hint`、deadline-aware NIC 带宽仲裁(见近期 commit)。

### 待后续梳理(TODO)
- [ ] TE 深入:RdmaContext/RdmaEndPoint、QP 管理、TENT 实现细节
- [ ] Store 架构:master service、client、replica/eviction/pin 机制
- [ ] EP/PG 的容错与恢复实现
- [ ] Python 集成层如何暴露给 SGLang/vLLM(connector)

---

## ③ TE 旧实现:RDMA 传输层(Context / EndPoint / WorkerPool)

代码位置:`src/transport/rdma_transport/`(6 文件,约 18 万行合计)。这是旧实现(`impl_`)最核心、最重的部分。

### 资源层级(一台机器视角)
```
RdmaTransport                                  ← "rdma" 协议的 Transport 实现
  └── context_list_: vector<RdmaContext>       ← 每张本地 RNIC(如 mlx5_0/1/…)一个
        ├── ibv_context / ibv_pd               ← 设备 & Protection Domain
        ├── MemoryRegionMap  (addr→ibv_mr)     ← 本 NIC 上注册的所有 MR(lkey/rkey)
        ├── cq_list_: vector<RdmaCq>           ← 多个 CQ,轮转分配
        ├── endpoint_store_: EndpointStore     ← QP 连接缓存(带淘汰)
        │     └── RdmaEndPoint (peer_nic_path) ← 到某个"远端NIC"的一组 QP
        │           └── qp_list_: vector<ibv_qp*>  (num_qp_per_ep 条 QP)
        └── worker_pool_: WorkerPool           ← 每个 Context 一组后台线程(post/poll)
```
关键概念:
- **RdmaContext = 一张本地 NIC 的全部资源**(device/pd/cq/mr/endpoints/worker)。多 NIC = 多 Context ⇒ 天然带宽聚合。
- **RdmaEndPoint = 本地NIC↔远端NIC 之间的一组 QP**。用 `peer_nic_path`(如 `192.168.3.76@mlx5_3`)标识。一个 endpoint 含 `num_qp_list`(默认 2)条 QP;每 QP 有独立 `wr_depth_list_` 深度计数。
- **NIC Path** = `server_name@device_name`,是选路/握手的基本寻址单位。

### EndPoint 连接与状态机
- 状态:`INITIALIZING → UNCONNECTED → CONNECTING → CONNECTED_WAIT_READY_ACK → CONNECTED`(可 → DESTROYING → DESTROYED)。只有 `CONNECTED`(readyToSend)才能 post WR。
- **握手**(交换 QP num/gid/lid,走 metadata 的 RPC 通道):
  - 主动方 `setupConnectionsByActive(peer_nic_path)`;被动方在 RPC 服务里 `setupConnectionsByPassive()`。
  - QP 状态迁移 INIT→RTR→RTS;启用 ready-ACK 时先进 `CONNECTED_WAIT_READY_ACK`,ACK 完成才转 `CONNECTED`。
- **两阶段销毁**(避免并发 submit 的 use-after-free):`beginDestroy`(置 active=false、QP→ERR 让硬件把 in-flight WR 冲刷到 CQ,不阻塞)→ `finishDestroy`(等 `wr_depth_list_` 全归零后真正 destroy_qp,超时 30s/重试 3 次兜底)。

### EndpointStore:QP 连接缓存 + 淘汰
- QP 连接昂贵,故对 endpoint 做**有上限缓存 + 淘汰**;两种策略:
  - `FIFOEndpointStore`:FIFO 淘汰。
  - `SIEVEEndpointStore`:NSDI'24 SIEVE 算法(类 clock + quick demotion),默认更优。
- 被淘汰/删除的 endpoint 进 `waiting_list_` 延迟回收(`reclaimEndpoint`,由 monitorWorker 周期驱动,不依赖新插入,避免故障下卡死,见 issue #1845)。

### WorkerPool:异步执行模型(核心)
每个 Context 一个 WorkerPool,后台多线程,绑定到 NIC 的 NUMA socket。**提交与执行解耦**:
```
submitTransferTask(切片) → context->submitPostSend(slices)
   → WorkerPool::submitPostSend:
       1. 逐 slice 用 selectPeerDevice() 选“远端NIC”(拿 dest_rkey + peer_nic_path)
          - segment desc 线程本地缓存(CONFIG_CACHE_SEGMENT_DESC,1s TTL)
          - 选中的 rail 若被 pause,则在远端其它 device 里找可用 rail
       2. 按 (target_id, device_id) 哈希分到 kShardCount(8) 个分片队列
          slice_queue_[shard][peer_nic_path].push_back(slice)
       3. notify 唤醒 parked 的 worker 线程
   ↓ 后台线程 transferWorker(thread_id) 主循环:
       - 空闲(submitted==processed 且无 outstanding CQ)→ park/cond_wait(1s)
       - performPostSend():从分片队列取 slice → endpoint->submitPostSend() → ibv_post_send 到 QP
       - performPollCq():poll CQ,WC 成功→slice->markSuccess();失败→重试/换 rail/markFailed
```
- **分片(kShardCount=8)**:降低多线程对提交队列的锁竞争(每分片独立 TicketLock)。
- **poll 与 post 可由不同线程承担**(`workerCanPost`/`workerCanPoll`),流水线化。
- **完成回传**:CQ 轮询到成功 → `slice->markSuccess()` 原子累加到 TransferTask/BatchDesc(接回 ②里的 batch 完成通知机制)。

### 容错 / 故障转移(WorkerPool 内)
- **Rail 级**:`rail_states_`(peer_nic_path→RailState),错误累计 ≥5 次则 pause 1s(`isRailAvailable` 期间自动改走其它 rail);`handlePathFailure` 统一处理路径失败(标记 rail、通知其它 worker、必要时删 endpoint)。
- **Context 级**:所有 rail 都不可用则累加 `context_failure_count_`,连续 32 次判本地 RNIC 硬件故障 → `context.set_active(false)`,整张网卡下线(选路时被跳过)。
- **Slice 级**:`shouldRetrySlice` 按 `max_retry_cnt` 重试;失败 target 短期(100ms)拉黑 `failed_target_ids`。
- **monitorWorker**:后台监控线程,处理设备异步事件(`doProcessContextEvents`)、区分"传输超时"与"poller 卡住"(靠 `last_poll_ts_ns_`)、驱动 endpoint 回收。

### 关键配置(globalConfig())
`slice_size`(切片粒度)、`fragment_limit`(尾片合并阈值)、`retry_cnt`、`max_wr`(QP 队列深度)、`num_qp_per_ep`(每 endpoint QP 数)、`max_wr * num_qp_per_ep`(提交 watermark)。

### 待后续梳理(TODO)
- [ ] TENT 新实现(`tent/`)与旧实现的差异

---

### ③补充:旧 RDMA 实现深入细节(精读 .cpp)

#### 线程模型(WorkerPool 构造)
- 每 Context 起 `workers_per_ctx`(默认 2,`MC_WORKERS_PER_CTX`)个 `transferWorker` + 1 个 `monitorWorker`,共 N+1 线程。
- **post/poll 分工**:`workers_per_ctx==1` 时单线程既 post 又 poll;否则 **线程 0 专职 poll,线程 1..N-1 专职 post**。post 线程按 `post_tid=thread_id-1`、步长 `N-1` 遍历 8 个 shard。
- 空闲(`processed==submitted` 且无 outstanding CQ)则 `parked_worker_count_++` 并 `cond_var_.wait_for(1s)`,提交端 notify 唤醒。

#### performPostSend(post 线程主体)
1. 按 shard 分配把 8 个分片队列的 slice(按 `peer_nic_path` 分类)搬入线程本地 `collective_slice_queue_[tid]`。
2. **redispatch 屏障**:若线程本地计数 < 全局 `redispatch_counter_`(被 `handlePathFailure` 自增),丢弃本地队列、对所有 slice 走 `redispatch`,强制刷新 endpoint 映射与 peer device 选择。
3. 按 `peer_nic_path`:`context_.endpoint()` 取/建 RdmaEndPoint;未连接则 `setupConnectionsByActive()`;未 ready 且 `readyAckTimedOut()` 则删 endpoint;然后 `endpoint->submitPostSend()`。

#### performPollCq(poll 线程主体)
- 每次每 CQ `ibv_poll_cq` 最多 `kPollCount=64` 个 WC。
- 成功→`slice->markSuccess()`,`processed++`;`IBV_WC_WR_FLUSH_ERR`→销毁中的正常 flush,直接 markFailed 不重试;其它错误→`handlePathFailure` + 按 `shouldRetrySlice` 决定进 `failed_slice_list`(→redispatch,递增 retry_cnt,达 `max_retry_cnt` 才最终失败)。
- 原子回收 `cq_outstanding_` 与各 QP 的 `wr_depth`。

#### RdmaEndPoint::submitPostSend(WR 构造 + 多 QP 限流)
- 每 slice 1 个 SGE:`addr=source_addr, lkey=source_lkey`;`wr_id=(uint64_t)slice`(poll 时识别);opcode=RDMA_READ/WRITE;`send_flags=IBV_SEND_SIGNALED`(每 WR 都 signaled);`remote_addr=dest_addr, rkey=dest_rkey`;链成链表。
- **多 QP 轮转**:遍历 QP,每 QP 分配 `ceil(remaining/remaining_qps)`,再 `min(assigned, qp_avail, cq_remaining)`。QP 可用深度 = `max_wr_depth_ - wr_depth_list_[qp]`;CQ 剩余 = `max_cqe - cq_outstanding_`。post 前先原子 `++wr_depth_list_[qp]`、`++cq_outstanding_`。
- `ibv_post_send` 失败:`bad_wr` 链上的 slice 移入 `failed_slice_list` 并回滚计数;之前成功的 WR 由硬件正常经 CQ 完成。

#### 握手与 QP 状态迁移
- `HandShakeDesc` 交换:`local_nic_path/lid/gid`、`qp_num`(每 QP 一个)、`ready_ack`/`ready_ack_supported`。走 metadata 的 RPC(`sendHandshake`)。
- 首个进入者置 CONNECTING(其它线程 spin+指数退避等,超时 10s);构造 desc→RPC 交换→`doSetupConnection` 迁移 QP。
- **QP 迁移**:RESET→INIT(`access = LOCAL_WRITE|REMOTE_READ|REMOTE_WRITE|REMOTE_ATOMIC`)→RTR(`path_mtu=min(activeMTU,cfg)`、dgid/sgid_index/dlid/dest_qp_num、`rq_psn=0`、`max_dest_rd_atomic=16`、`min_rnr_timer=12`)→RTS(`timeout=14`、`retry_cnt=7`、`rnr_retry=7`、`max_rd_atomic=16`)。可选 `mlx5dv_modify_qp_lag_port`(LAG 绑定)、`mlx5dv_modify_qp_udp_sport`(RoCEv2 UDP 源端口)。
- **ready-ACK**:两端到 RTS 后先 `CONNECTED_WAIT_READY_ACK`,主动方 `sendReadyAck()` 再发一轮 RPC,被动方收到后升 `CONNECTED`(超时 10s 则删 endpoint)。peer 不支持则直接 CONNECTED。
- **RTR 阶段 EINVAL** 触发 Auto GID 重试(默认 2 次)。

#### 内存注册与 rkey/lkey 查询
- `registerMemoryRegionInternal`:CPU 内存 `ibv_reg_mr`;CUDA 内存优先 `WITH_NVIDIA_PEERMEM` 否则走 `cuMemGetHandleForAddressRange` 拿 dmabuf fd → `ibv_reg_dmabuf_mr`;HIP 类似(`hsa_amd_portable_export_dmabuf`)。超 `max_mr_size`(默认 1TB)截断。
- `memory_region_map_`:`map<uintptr_t, {addr, ibv_mr*}>`;`rkey/lkey(addr)` 用 `upper_bound` 找前一 entry 再判范围,RWSpinlock 读锁。

#### GID 选择(findBestGidIndex)
- 只接受 `ROCE_V2` 和 `IB` 类型。7 级优先(高→低):有网卡且 RoCEv2/IB(最优)→无网卡 RoCEv2/IB→有网卡私有 IPv4→有网卡但 overlay/link-local(降级)→无网卡私有 IPv4→无网卡降级→非零兜底。识别 overlay(calico/flannel/cni/docker/vxlan)并降级。同级按 gid_index 升序。

#### monitorWorker / 异步事件
- 每秒:`reclaimEndpoints()`(回收 waiting_list,修复 #1845)、CQ 停滞诊断(poll gap≥5s、transfer≥30s 告警)、`epoll_wait` 监听 ibv 异步 fd。
- `doProcessContextEvents`(先 ack 再动 endpoint,防死锁):`QP_FATAL`→删该 endpoint;`DEVICE_FATAL/CQ_ERR/WQ_FATAL/PORT_ERR/LID_CHANGE`→`set_active(false)`+`disconnectAllEndpoints`;`PORT_ACTIVE`→`set_active(true)`+重置故障计数。

---

## ④ TENT 新实现(next-gen Transfer Engine)

代码位置:`mooncake-transfer-engine/tent/`(~6.8 万行,独立命名空间 `mooncake::tent`)。旧实现的重写版,由 `use_tent_` 开关在门面 `TransferEngine` 里二选一。设计更模块化,核心新增能力:**QoS / 优先级 / SLO deadline 感知调度**、**配置驱动的 transport 选路策略**、**事件驱动 progress worker**。

### 目录结构(职责清晰分层)
```
tent/
  include/tent/ , src/
    transfer_engine.{h,cpp}       ← C ABI(tent_* 函数)+ C++ 门面
    runtime/                      ← 运行时核心(见下)
    transport/                    ← 各协议:rdma/ tcp/ nvlink/ mnnvl/ shm/ gds/
                                     io_uring/ ascend/ sunrise_link/ tpu/ bufio/ fault_proxy
    metastore/                    ← etcd / redis / http(元数据后端)
    platform/                     ← cpu / cuda / rocm / ascend / sunrise / tpu(设备抽象)
    metrics/                      ← 指标(config_loader + tent_metrics)
    rpc/ , common/(config/status/types + concurrent 无锁原语)
  config/                         ← transfer-engine.json / cluster-topology.json(配置驱动)
  plugins/                        ← cuda / rocm 设备插件(动态加载)
```

### runtime 核心组件(`src/runtime/`)
| 组件 | 职责 |
| --- | --- |
| `TransferEngineImpl` | 运行时总装。持有 admission queue、transport selector、segment manager、progress worker、proxy manager 等。|
| `LocalTransferAdmissionQueue`(admission_queue) | **准入 + 派发调度**。管 outstanding owner/bytes 上限、staging 预留;支持 deadline-aware EDF 排序、deadline 邻近提升、不可行丢弃。|
| `TransportSelector`(transport_selector) | **配置驱动选路**。按规则(segment_type/priority/memory pattern/size/devices/intent_type)从策略表选 transport 优先级列表。无配置则回落默认(File: GDS→IOURING→RDMA;Memory: 按 buffer_transports)。|
| `ProgressWorker`(progress_worker) | **事件驱动推进**(issue #2116)。transport 完成后 `notifyBatchMaybeReady` 唤醒,驱动 `progressBatch` 单步推进;另有低频 fallback tick 兜底。|
| `SegmentManager`/`SegmentRegistry`/`SegmentTracker`/`Segment` | Segment(注册内存/文件)的注册、发现、生命周期、跟踪。|
| `ProxyManager` | CPU-proxy 路径(异构/中转)。|
| `ControlPlane` / `MetaStore` | 控制面 + 元数据后端(etcd/redis/http)。|
| `Slab` / `MemoryProber` / `Topology` / `Platform` / `TransportLoader` | 内存池、内存类型探测、拓扑、设备平台、transport 动态加载。|

### 关键数据结构与新概念
- **`Request`**(types.h)较旧版新增字段:
  - `priority`:PRIO_HIGH(0)/MEDIUM(1)/LOW(2) —— QoS 优先级。
  - `transport_hint`:`UNSPEC=跟随策略`,否则 pin 到指定 transport(RDMA/MNNVL/SHM/NVLINK/GDS/IOURING/TCP/AscendDirect/SUNRISE_LINK/TPU)。
  - `deadline_ns`:绝对 steady_clock 纳秒 SLO 截止时间(0=无)。
  - `intent_type`:业务意图,用于策略匹配 —— `FOREGROUND_GET / BACKGROUND_PREFETCH / MIGRATION / CHECKPOINT / WEIGHT_LOADING / STAGING_INTERNAL`。
  - `policy_name`:可选,绑定到具体命名策略。
- **`TaskInfo`**(impl):每 task 记录 transport type、failover_count、device_mask(quota 分配)、qp_pool(命名 QP 池,RFC #2568)、staging 标志、以及三个时间戳(submit/dispatch/post,用于 SLO 度量)。
- **QoS 与 SLO 是渐进式 opt-in**(通过 RFC 分步落地,默认关闭保持旧行为):
  - RFC **#2519**:deadline 感知。step1 仅观测(完成时记 MLU=实际传输时间/可用窗口);step2 `deadline_aware` 开启 EDF 排序;step3 `mlu_local_threshold` 不可行则丢弃并触发 `on_local_decode_suggested`(上层 vLLM/SGLang 本地重算/压缩)。另有 `promotion_slack_ns` deadline 邻近提升。
  - RFC **#2568**:命名 QP 池布局。
  - deadline-aware NIC 带宽仲裁(`transport/rdma/bw_arbitration.h`,近期 commit `[TENT]`)。

### 主要调用流程(submitTransfer)
```
submitTransfer(batch, requests, notifi?, owner_kind)
  1. retainBatch → prepareSubmit(切分/规划 tasks → PreparedSubmit)
  2. shouldQueueSubmit? (受 QoS/admission 管控)
       是 → enqueuePreparedSubmit(进 admission queue)
            → refillDispatchWindow(按容量/EDF/deadline 选 owner 派发)
            → notifyRuntimeQueueReady(唤醒 progress worker)
       否 → commitPreparedSubmit(直接提交 transport,旧式快路径)
  3. notifi? → addSubmitHook(完成时发通知)
  ↓ 推进由 ProgressWorker 事件驱动:progressBatch 单步 → 各 transport post/poll
     失败 → failover(TaskInfo.failover_count,按 selector 的 transport 优先级换路)
```
- 相比旧实现的 WorkerPool "提交即入队、后台线程 post/poll",TENT 多了一层 **admission queue(准入+QoS 派发)**,并把推进抽象成可单步的 `progressBatch` 状态机(支持 `cancelTransfer` 尽力取消)。

### 与旧实现的核心差异
| 维度 | 旧实现(`src/`) | TENT(`tent/`) |
| --- | --- | --- |
| 选路 | 硬编码 + topology 亲和 | **配置驱动策略表**(transfer-engine.json)+ intent/priority/size 匹配 |
| 调度 | 无准入,直接切片入 worker 队列 | **admission queue**:outstanding 上限、staging 预留、EDF/deadline 派发 |
| QoS/SLO | 无 | priority / deadline_ns / intent_type / MLU 可行性 |
| 推进模型 | 每 Context 常驻 worker 线程池 | **事件驱动 ProgressWorker** + 可单步 progressBatch |
| 取消 | 无 | `cancelTransfer`(尽力) |
| 命名空间/接口 | `mooncake::` C++ 类 | `mooncake::tent::` + 独立 `tent_*` C ABI + pybind |
| 元数据/平台 | 插件式 metadata | metastore(etcd/redis/http)+ platform 抽象 + 动态 transport 加载 |

### ④补充:TENT 深入细节(精读 runtime + transport/rdma)

#### 准入队列 LocalTransferAdmissionQueue(单线程,同步由 impl 持有)
- **tryAdmit 四道上限**(任一超出即 `TooManyRequests`):总 owners、总 bytes、User owners、User bytes。后两者上限 = 总上限 − staging 预留(`staging_owner_reserve`/`staging_byte_reserve`),即给 `StagingInternal` 流量留专用配额。
- 另拒绝:`batch_token==0`、`length==0`、public_key 重复、超 `batch_slots_left`、跨 submit 重复提交同 task。
- 成功时:`deadline_aware` 则按 `deadlineKey`(deadline_ns=0 映射 UINT64_MAX 排最后)用 `upper_bound` **插入 EDF 有序位置**,否则 FIFO push_back。

#### pickForDispatch(派发选择)
- FIFO 或 EDF(admit 时已排好,派发只需队首消费)。
- **promotion_slack_ns 邻近提升**:派发前 `stable_partition` 把 `(deadline−now)<slack` 的紧迫 owner 整体提到队首(保持段内 EDF 序)。
- **mlu_local_threshold 不可行丢弃**(需 deadline_aware + bandwidth_provider):`MLU = (length/bw_bps) / ((deadline−now)/1e9)`,`MLU≥阈值` 则丢弃(状态置 CANCELED、扣 outstanding、触发 `on_local_decode_suggested` 让上层本地重算/压缩)。丢弃不占派发预算,继续扫下一个。
- 正常派发:若 `length>remaining_bytes` 则整体 break(EDF 序,后面更满足不了)。

#### TransportSelector::select(配置驱动选路)
- 按 JSON 顺序遍历 policy,首条全匹配胜出。`matchesPolicy` AND 检查:policy_name、segment_type、same_machine、local/remote memory pattern(支持 `*`)、min/max size、priority(精确)、intent_type。
- 候选 transport = 命中 policy 的 `.transports`(空则回落 buffer_transports)。有 `hint` 则 `reorderWithHint`(hint 不在候选→UNSPEC→FAILED,否则提到第 0 位)。
- 逐个 `isTransportAvailable` 校验:NVLINK/SHM/TPU 仅 same_machine;按 mem 类型组合查 capabilities(dram_to_dram/gpu_to_gpu/...)。`transport_index=failover_count` 选第几个可用。
- 无 policy 默认:file=[GDS,IOURING],memory=回落 buffer_transports。legacy 模式绕过 selector 走硬编码。

#### submit 全流程(impl)
- `prepareSubmit`:校验 hint;开启 `merge_requests_` 则合并连续同 opcode/target/hint/地址的 request(生成 task_lookup 映射);对合并后 request `resolveTransport(index=0)`;TCP+需 staging 则标记 `owner.staging`。**不改 batch,只产出 PreparedSubmit**。
- `shouldQueueSubmit`=true 条件:`runtime_queue` 开启 且(owner_kind 是 StagingInternal 或本次无 staging owner)。含 staging 的普通 submit 走直接 commit 快路径。
- 走队列:`enqueuePreparedSubmit`(分配 batch_token、校验 `length≤max_dispatch_bytes`、`tryAdmit`、插 TaskInfo)→ `refillDispatchWindow` → `notifyRuntimeQueueReady`。
- `refillDispatchWindow`:持 `progress_mutex_`,查 inflight 上限,`pickForDispatch` 取一批,逐个 `dispatchQueuedOwner`(派发时刻**重新** resolveTransport;UNSPEC→FAILED;TCP+staging→staging_proxy;否则分 SubBatch 调 `transport->submitTransferTasks`)。
- `commitPreparedSubmit` 快路径:按 transport 分类直接下发,设 `device_mask`/`qp_pool`,`attachProgressNotifier` 挂完成回调。

#### progressBatch 状态机(单步,`getBatchStatus(allow_failover=true)`)
- 持锁→`refillDispatchWindow`→遍历非 derived task:仍在队列等待则跳过;非 PENDING 直接累计;否则 `pollTaskStatus`(staging→proxy;UNSPEC→FAILED;正常→`transport->getTransferStatus`)→`updateTaskStatusAfterPoll`。
- **failover**:仅当 `allow_failover && !cancel && status==FAILED && type!=UNSPEC` → `resubmitTransferTask`:`++failover_count`(超 `max_failover_attempts_`=3 则终止)、`xport_priority=failover_count` 作 `transport_index` 换下一个候选 transport 重新分 SubBatch 提交、置回 PENDING。
- 终态:全成功→COMPLETED;全终态且有失败→worst_failure;否则 PENDING。COMPLETED 触发 submit hooks(通知)。`lazyFreeBatch` 等 `runtime_refs==0` 且全终态才真正释放。

#### ProgressWorker 事件循环(issue #2116)
- `notifyBatchMaybeReady` 用 `unordered_set queued_` 去重后入 `order_` 队列 + `notify_one`。
- runner:有活跃 runtime queue 且 `fallback_interval>0`(默认 50ms)则 `wait_for` 超时→fallback tick(无条件 `progressRuntimeQueue`,兜底那些无 completion-wake 回调的 transport);否则纯 `wait`。取出后调 `progressBatch`/`progressRuntimeQueue`+`lazyFreeBatch`。

#### cancelTransfer(尽力取消)
- **可取消**:仍在 admission queue 未进 dispatch window(`cancelQueuedOwner`,状态须为 Queued);`type==UNSPEC` 的 PENDING task。
- **有条件**:已在 dispatch window→仅当 `transport->supportsCancellation()` 调 `cancelTransferTask`,否则 NotImplemented。
- **不可**:staging 任务;已终态→OK(no-op)。合并的派生 task 一并置 `cancel_requested`。

#### TENT RDMA transport 关键差异(vs 旧 rdma_transport)
- **完全 partitioned 的 worker**:每 worker 自带 3 条优先级 `BoundedMPSCQueue<...>`(HIGH/MED/LOW,无锁 MPSC,容量 8K)、独占 CQ(`tl_wid % num_cq_list`)、独占 RailMonitor —— **无锁竞争**(旧版是共享 8 分片队列 + TicketLock + 全局 posted_slices + redispatch_counter)。提交按最小 inflight 选 worker。
- **DeviceSelector 智能选路(quota.cpp)**:默认 EWMA 预测,`predicted_time=(inflight+slice)/ewma_bw` + NUMA 层权重 + 抖动,选分最低设备;多 slice 按反比权重分摊,每 100 次一次探测采样防 EWMA 饥饿;`alpha=0.01`。`device_mask` 位掩码过滤可用 NIC。
- **跨进程共享配额(shared_quota.cpp)**:`shm` 中全局槽计数器(2ms 转),3 槽(仅HIGH / HIGH+MED / 全部)轮转 → 基于时间的跨进程 QoS;`canSend(priority)` 门控;`PTHREAD_MUTEX_ROBUST` 容进程崩溃。
- **带宽仲裁(bw_arbitration)**:同优先级层内竞争同一 NIC 的 flow 按 MLU 降序发(紧迫先发,稳定排序保 FIFO)。priority=垂直层级,BW 仲裁=水平层内排序,deadline 仅影响顺序不影响准入。opt-in `deadline_bw_arbitration`,默认 FIFO。
- **RailMonitor**:每 `(local_nic,remote_nic)` 对独立 state,**指数退避**(30s→上限 300s)+ 滑动错误窗口(10s)+ 阈值 3;`markRecovered` 有 O(1) 无锁快路径;NUMA 感知 best_mapping;自动拓扑匹配(同名>CUDA 直连>同 NUMA 最小负载>任意)。旧版仅单 `error_count+pause_until` 无退避无窗口全局锁。
- **QP 池(RFC #2568)**:命名 `QpPoolSegment` 把 endpoint 的 QP 切成不重叠区间,每池可独立 `service_level`/`traffic_class`;`computeQpPoolSegments`(纯函数,空配置退化旧行为)+ `selectQpInPool` 把流量锁在 `[begin,begin+num_qp)`;底层 QP 编号位置配对,BootstrapDesc 字节兼容。
- **优先级提升(promotion_policy.h)**:防低优先级饥饿。默认 `HeadOnly`(只看队首 enqueue_ts,超时则整队提升,缺陷:非饥饿项被连带);opt-in `PerEntry`(#2528,逐项判超时只提超时项,且每 tick 同时处理 MED→HIGH 与 LOW→MED)。默认超时 10ms,~1ms 检查一次。
- **EndpointStore**:仍是 FIFO / **SIEVE**(默认,NSDI'24 clock+快速降级,visited 标志 + hand 指针扫描),`reclaim` 每秒由 monitor 处理 `weak_ptr::expired`。
- **Slice/Task 结构**:`RdmaTask` 用原子 `status_word`+`success/resolved_slices`+`ref_count`(UAF 保护)聚合;`RdmaSlice` 携带非拥有 `rail_monitor*`(热路径免字符串查表)、`quota_charged`(防重复释放)、`enqueue/submit_ts`、`priority`。完成聚合用 CAS 只处理 PENDING→终态。
- **notification QP**:每 endpoint 一条基于 SEND/RECV 的带外控制通道(64KB、256 槽、wr_id 流控)。
- **两阶段销毁**:`beginDestroy`(QP→ERR 刷硬件)/`finishDestroy`(等 CQ 耗尽再销毁),endpoint 生命周期严格单向(失效不重用)。

### 待后续梳理(TODO)
- [ ] Mooncake Store 架构

---

## ⑤ Mooncake Store(分布式 KV 缓存存储引擎)

代码位置:`mooncake-store/`。设计文档:`docs/source/design/mooncake-store.md`。定位:**面向 LLM 推理的分布式 KV cache**(区别于 Redis/Memcached:key 非 value 派生,Put 后不可变直到删除,强一致 Get)。构建在 Transfer Engine 之上。

### 总体架构:两大组件
```
   ┌─────────────┐   RPC(coro_rpc)  ┌──────────────────────┐
   │   Client    │ ───元数据请求───→ │   Master Service     │
   │(应用侧+存储侧)│ ←──replica_list── │(只管元数据,不碰数据流)│
   └──────┬──────┘                   └──────────────────────┘
          │ 数据面:Client↔Client 直连(经 Transfer Engine 零拷贝),绕开 Master
          ▼
   其它 Client 的 segment 内存
```
- **Master Service**:独立进程,编排整个集群的逻辑存储池、管理节点 join/leave、对象空间分配与元数据、淘汰策略。**只提供元数据,不经手任何数据流**。注意:TE 需要的 metadata service(etcd/redis/http)与 Master 是分开部署的。
- **Client**:唯一的客户端类,但**双重角色** —— ①作为 client 发 Put/Get;②作为 store server 贡献一段连续内存给分布式缓存池,供别的 Client 读写。`global_segment_size=0`→纯 client;`local_buffer_size=0`→纯 server。
- **三种使用方式**:embedded(与 vLLM 同进程)、embedded dummy-real(每 rank 一个无资源 dummy,每 instance 一个 real 持资源,TP=8 → 8 dummy + 1 real)、standalone store service。

### Put 三段式协议(防脏读)
```
PutStart(key,length,config,slice_lengths) → Master 按 ReplicateConfig 分配空间
   → 返回 replica_list(每个含 buffer 地址/segment/transport endpoint)
数据面:Client 用 Transfer Engine 并行写数据到各 segment(striping)
PutEnd(key) → Master 把 replica 状态标 COMPLETE,对象变为可读
```
- **多副本分散**:保证同一对象每个 slice 的多副本落在**不同 segment**(allocation_strategy 用 `used_segments` set 去重);不同对象的 slice 可共享 segment。best-effort:空间不足则尽量多分。
- Get:`GetReplicaList(key)` 拿副本 → 选最优副本 → TE 从远端 client 零拷贝读入本地已注册 buffer。

### Master 侧关键实现(精读 master_service)
- **元数据分片**:`metadata_shards_[1024]`,每片 `SharedMutex + unordered_map<key, TenantState>`;路由 `hash(tenant_id, user_key) % 1024`。`ObjectMetadata` 内含 SpinLock 保护 lease/soft_pin_timeout。**锁顺序**(防死锁):client_mutex > tenant_quota_policy > snapshot > metadata_shard > tenant_quota_shard > segment_mutex。
- **分配**:`AllocatorManager`(segment_name→allocator 列表),allocator 类型 Cachelib 或 OffsetAllocator。`AllocationStrategy` 多态:Random / FreeRatioFirst / CXL / SSD_FreeRatioFirst / LOCAL_FIRST。`FLEXIBLE_DUAL_REPLICA`(mem 或 NoF 有一个成功即可)vs `RELIABLE_MULTI_REPLICA`(严格全满足)。
- **淘汰**:实际走 `BatchEvict` 两阶段(非 LRUEvictionStrategy 类):①16 线程并行扫描收集 candidate,按 `lease_timeout` 排序;②逐个 evict。**第一遍跳过 soft-pin,不够再第二遍允许淘汰 soft-pin**;`hard_pinned` 永不淘汰。`CountMinSketch`(4×4096)仅用于 promotion-on-hit 频率门控(阈值默认 2)。NoF(SSD)副本有独立 `NoFBatchEvict`。
- **Replica 状态机**:`UNDEFINED→INITIALIZED→PROCESSING→COMPLETE→REMOVED/FAILED`。变体 variant:Memory / NoF / Disk / LocalDisk。`refcnt_` pin 住源副本(offload/promotion 时)。
- **副本选优**(replica_selection):本地MEM > 本地NOF > 远程MEM(启用 `MC_STORE_REPLICA_SCORING` 时按 scorer 打分:rdma 0.0 < tcp 1.0 < unknown 2.0)> 远程NOF > LOCAL_DISK > DISK。
- **故障检测**:Client `Ping` 入无锁队列(128K);`ClientMonitorFunc` 1s 轮询,`client_live_ttl`(默认 10s)超时则 PrepareUnmount 其所有 segment、ClearInvalidHandles、CommitUnmount。NoF 独立 100ms 心跳(SPDK probe,连续 3 次失败自动卸载)。
- **HA 模式**:`StandbyStateMachine`(STOPPED→CONNECTING→SYNCING→WATCHING→PROMOTING→PROMOTED)+ etcd 选主 + `OpLogReplicator` 跟随复制;Promote 时 gap resolve + 最终追赶(≤30s/100 批)。Leader 选举在 `MasterServiceSupervisor`。
- **快照**:`MasterSnapshotManager` 用 **fork() COW** 子进程序列化(msgpack)元数据/segment/task_manager,默认 10min 间隔,子进程超时 SIGKILL;重启按 catalog 最新快照恢复,失败则空状态启动。
- **租户配额**:`tenant_quota_shards_[1024]`,reserve(PutStart)→commit(PutEnd)→abort(revoke/失败)三阶段;超额时单租户淘汰后重试(≤2 次)。默认 `enable_multi_tenants=false`。

### Client 侧与数据面关键实现(精读 client_service/real_client/transfer_task)
- **RealClient vs DummyClient**:`RealClient` 内含真正的 `Client`(持 MasterClient/TransferEngine/TransferSubmitter),既是 API 门面又是存储 server。`DummyClient` 是轻量代理,无 TE/无 MasterClient/无 segment,通过 **memfd+mmap 共享内存** 与同机 RealClient 共享地址空间,经 coro_rpc + UDS 传 FD 转发所有操作。RealClient 维护 `dummy_base_addr→real_addr` 映射做 VA 转换。Dummy 每 1s ping,断连则重连重注册。
- **Get 数据流**:`Query`(MasterClient.GetReplicaList,带 lease_ttl)→ `SelectBestReplica` → 热缓存重定向(RedirectToHotCache 原地改描述符指向本地 HotMemBlock)→ `TransferRead` → `TransferSubmitter::submit`。**零拷贝**:用户 Slice 指向已注册内存,TE 直接 RDMA:远端 segment→用户 buffer。
- **Put 数据流**:PutStart 拿分配 → 各副本**并行** TransferWrite(每副本一个 TransferFuture;`prefer_alloc_in_same_node` 合并同 segment 传输)→ `DetermineFinalizeDecision` → 成功副本 PutEnd / 失败副本 PutRevoke。大对象按 Slice 切分成多 TransferRequest 并行;`global_segment_size` 超 `max_mr_size` 时按块切分注册(striping)。
- **传输策略选择**(`selectStrategy`):本地端点+`MC_STORE_MEMCPY`→LOCAL_MEMCPY(MemcpyWorkerPool,GPU 感知);远程→TRANSFER_ENGINE(openSegment+submitTransfer);LocalDisk→FILE_READ;NoF→SPDK_NVMF。
- **内存池**:`ClientBufferAllocator` 管一大块连续注册 buffer,内部委托 `OffsetAllocator`(bin-based,32 top×8 leaf=256 空闲链,受 Sebastian Aaltonen 启发,64 位寻址,可序列化)。`PinnedBufferPool`(默认上限 32)缓存固定 buffer 用于 GPU D2H 暂存。
- **多级存储 RAM(L1)→SSD(L2)**:由 Master 心跳驱动。**Offload**:Master 标记 `offloading_objects`→`FileStorage::OffloadObjects`(PinnedBufferPool 暂存,必要时从 GPU 读)→`storage_backend->BatchOffload`→NotifyOffloadSuccess 标 LOCAL_DISK 副本→逐出内存。**Promotion**(按需):Get 命中 LocalDisk-only key→Master 入队 `promotion_objects`→`FileStorage::ProcessPromotionTasks`(PromotionAllocStart 暂存 PROCESSING 副本→BatchLoad 读盘→PromotionWrite→NotifyPromotionSuccess 转 COMPLETE)。后端:`BucketStorageBackend`(bucket 文件≤256MB/500key,O_DIRECT 对齐,FIFO/LRU)、`OffsetAllocatorStorageBackend`(单文件紧打包,元数据 1024 分条)。
- **TransferSubmitter**(transfer_task):Store 读写→TE batch 的桥。`submitTransferEngineOperation`:openSegment→逐 Slice 建 `TransferRequest`(source=本地注册buffer,target_offset=buffer_address+offset)→allocateBatchID→submitTransfer→TransferFuture 等完成。`submit_batch` 合并同端点多 key 到一个 batch。
- **Copy/Move 任务**:Master 侧异步任务(task_manager),Client `TaskPollThreadMain` 定期 `FetchTasks(16)` 拉取→`ExecuteReplicaTransfer`(CopyStart 分配 target→零拷贝切 source→逐 target TransferWrite→CopyEnd/Revoke)。Move 额外删 source。
- **RPC 框架**:`ylt/coro_rpc`(C++20 协程)。`MasterClient` 持 `coro_io::client_pools`,`invoke_rpc<&WrappedMasterService::PutStart>(...)` 模板调用;`MC_RPC_PROTOCOL=rdma` 启用 ib_socket。HA 下 `LeaderCoordinator` 监控 leader 变更自动重连。

### 待后续梳理(TODO)
- [ ] Mooncake EP / PG(MoE 容错执行)
- [ ] Python 集成层(SGLang/vLLM connector)如何调用 Store/TE

---

## ⑥ Mooncake EP / PG(容错 MoE 分布式执行)

代码位置:`mooncake-ep/`(CUDA/kernel)、`mooncake-pg/`(PyTorch 后端)。文档:`docs/source/design/mooncake-{ep,backend-pg}.md`、`docs/source/python-api-reference/ep-backend.md`。构建开关 `-DWITH_EP=ON`,随 CUDA wheel 分发。二者关系:**先建 Mooncake PG(process group),再从该 PG 构造 EP `Buffer`**;PG 用于常规集合通信 + EP bootstrap 元数据交换。核心卖点:**在部分 rank 失效时不重启整个推理服务继续 MoE 服务**。

### 分工
```
   torch.distributed
        │
   Mooncake PG(mooncake / mooncake-cpu 后端)  ← 集合通信 + P2P + 弹性恢复
        │  提供 get_active_ranks / get_preferred_hca / recover_ranks / join_group
        ▼
   Mooncake EP(expert-parallel dispatch/combine)  ← MoE token 路由 + 结果聚合
        │  两者数据面都直达
        ▼
   Transfer Engine(PG 复用) / Device API(EP 直接 GPU-initiated:NVLink P2P + IBGDA)
```

### Mooncake EP(专家并行运行时)
- **定位**:遵循 **DeepEP low-latency 编程模型**,但把底层 NCCL/NVSHMEM GIN 传输**换成 Mooncake Device API**(`comm_device.cuh` 的 `mc_route_put`/`mc_rdma_put`/`mc_signal`/`mc_red_add`)。分两版:legacy `MooncakeEpBuffer` 与移植自 DeepEP official 的 `MooncakeElasticBuffer`(拓扑感知/TMA/hybrid)。
- **dispatch**(token→expert 路由):每 rank 持 `[num_tokens,hidden]` + `topk_idx`(每 token 选 top-k 全局 expert)。按 topk_idx 把 token 发到目标 expert 所在 rank 的接收槽,接收方按 local-expert-major 打包成 `packed_recv_x`;含 FP8 量化(per-128-channel amax→scale_inv)。
- **combine**(结果聚合):各 local expert 输出沿源路由反向发回原 token 所在 rank,按 `topk_weights` 加权求和(`combine_reduce_epilogue`,float 累加转 BF16)。支持 zero_copy(直接写 GDR buffer)。
- **传输方式**:cooperative launch + grid barrier;NVLink 走 peer-mapped VA 的 warp-cooperative load/store;跨节点走 **IBGDA**(mlx5gda,GPU 直发 RDMA WRITE/atomic,无 CPU 介入)。**不用 NVSHMEM,也不走 TE 的 CPU submitTransfer 路径**;TE 的 P2p/Rdma transport 仅用于建链(MR 注册/IPC handle/QP/peer 地址表)。
- **buffer**:GDR buffer 经 `P2pTransport::allocateBuffer` 分配(NVLink+IBGDA 共用),`BufferPair` 双缓冲(乒乓),每套含 send/recv 各自的 signal+data 区。`get_ep_buffer_size_hint` 按 `num_experts × num_max_dispatch_tokens_per_rank × (2×int4 + hidden×bf16)` 估算。

#### EP 容错核心:active_ranks
- `active_ranks` 是 `[num_ranks]` int32 GPU tensor,传入 dispatch/combine。接收等待循环轮询 signal buffer,**超时则 `active_ranks[src_rank]=0` 标记失效并停止等待**(不再收该 rank token)。
- combine 的 reduce 循环**跳过 `active_ranks[expert_src_rank]==0` 的 expert 贡献**,结果由剩余 expert 的加权和构成。
- 失效 rank 的 token:dispatch 侧相当于丢弃(上层需感知 active_ranks 不再往失效 expert 派 token)。切换活跃 rank 后需 `sync_ibgda_peers`/`sync_nvlink_ipc_handles` 刷新传输层。

#### Elastic(弹性)
- `MooncakeElasticBuffer` 内含一个 `MooncakeEpBuffer`,额外:拓扑感知(区分 `num_scaleup_ranks` NVLink 域 / `num_scaleout_ranks` RDMA 域)、hybrid 模式(scaleout→scaleup 内 NVLink 转发)、CPU 同步精确裁剪输出、cached 模式复用 handle、expand 模式(输出行=tokens×topk)、TMA 异步拷贝、deterministic 模式。
- **弹性是有限组合的静态模板 kernel**:`num_experts`/`hidden`/`topk` 等作 kernel template 参数,运行时 switch 匹配预编译组合,不匹配则抛 `unsupported_elastic_config`。
- `ElasticNativeHandle` 关键字段:`psum_num_recv_tokens_per_expert`(expert 维前缀和切分输出)、`recv_src_metadata`(combine 回溯 token 来源)、`dst_buffer_slot_idx`(每 token 分到的接收 slot)、`channel_linked_list`(hybrid 下 scaleup 域内 token 流转路径)。

### Mooncake PG(PyTorch ProcessGroup 后端)
- **注册**:`.so` import 时用 `__attribute__((constructor))` 调 `Backend.register_backend("mooncake"/"mooncake-cpu")`。**`MooncakeBackend` 继承 `c10d::ProcessGroup`(非 Backend)**;P2P 分发路径需要 Backend,故建 `MooncakeP2PShim`(继承 Backend,委托 send/recv)注册进 `deviceTypeToBackend_`。
- **MooncakeBackendOptions**:`activeRanks_`(int32 mask,GPU/CPU 与后端一致,暴露给 kernel 与 host)、`isExtension_`(替换/加入进程,初始为 local-only 待 `joinGroup` 才发布元数据)、`maxWorldSize_`(预留容量,多余 slot 标 inactive 供预轮询未来加入者)。
- **集合通信**(all_reduce/all_gather/broadcast/reduce_scatter/...):由 `MooncakeWorker` 承载。切 chunk→`tensorToBuffer`(cudaMemcpyAsync 进注册的 send_buffer)→host worker 线程构 `TransferRequest` 列表→`engine_->submitTransfer` 批量 RDMA/NVLink 写→发/等 1-byte sync 信号→`reduceKernel`(GPU,SUM/MIN/MAX/PROD,BF16 走 float 累加)写回用户 tensor。DUAL buffer 乒乓;GPU 侧 `enqueueTaskKernel` spin-wait 等 host 完成。
- **底层传输**:**PG 复用 TransferEngine**(进程级 singleton,可经 `setExternalEngine` 注入 EP 的 engine)。所有数据移动走 `submitTransfer`,TE 内部选 RDMA/TCP/NVLink。

#### PG 容错核心
- **ConnectionPoller**(单例后台线程):轮询各 peer 状态机(WAITING_STORE→...→CONNECTED),CONNECTED 后检查 `peerConnected`/`global_peerConnected_` 一致性,任一为 false 则清元数据、reset、回 WAITING_STORE 重连。
- **集合通信内检测**:worker 线程在传输阶段若 `getTransferStatus==FAILED`,或超 **100µs ping 超时** 且 `probePeerAliveByID` 非 Alive,则置该 peer `peerConnected/activeRanks=false`(**静默标记,不抛异常**,collective 跳过失效 peer 数据继续)。
- **P2P 检测**:P2PProxy 传输/credit/ack 超时(默认 30s)→`reportBrokenPeer`→reset + epoch++。
- **上报**:失败只更新 `activeRanks` mask,用户经 `get_active_ranks` 查询;失败经 `global_peerConnected_` 传播给同进程其它 backend。

#### 弹性恢复(rank recovery,不重启服务)
- **健康侧**:`getPeerState` 用 **MIN all_reduce** 让所有健康 rank 对"可恢复 rank 列表"达成一致;确认候选已连接后 `recoverRanks` 置 `activeRanks[rank]=true`、更新 activeSize、把 `ExtensionState`(activeRanks/epoch/taskCount)经 Store 下发。
- **加入侧**:以 `is_extension=True`+`max_world_size` init→local-only→`joinGroup`(发布元数据、注册 poller、`waitUntilAllConnected`、从 Store 读 ExtensionState 恢复状态)。
- **Epoch**:P2P 每 peer 原子递增 epoch,控制槽携带 epoch,接收方丢弃 epoch 不匹配的槽,避免旧连接残留干扰。Store 作为弹性元数据交换点(server_name/buffer/extension_state key)。

#### Worker 模型 & P2P
- `MooncakeWorker`(每 device 一个,多 backend 共享):驱动 collective,worker 线程 poll 4 个 task slot 的三段状态机(TRANSFERRED_1→SIGNALED_1→DONE)。`putTaskCpu`(future 递归 chunk)/`putTaskCuda`(enq_stream + GPU spin-wait,返回 torch::Event)。
- **P2PProxy**(send/recv/batch_isend_irecv 走此路,独立于 collective):**credit-based RDMA pull** —— 接收方主动分配 chunk 写 CreditSlot 给发送方,发送方 poll credit→拷 staging→RDMA 写 RecvPool→写 AckSlot;接收方 poll ack→拷回用户 tensor。chunk 默认 16MiB,控制槽 header/footer 双 token 防 RDMA 部分写撕裂。P2PDeviceWorker 每 device 共享 send/recv 线程 + 固定 SendPool/RecvPool(默认各 128MiB)。

### EP↔PG↔TE 三者关系
- **EP 用 PG bootstrap**:EP 初始化建 mooncake PG,经 `set_transfer_engine` 把自己的 TE 注入 PG;PG 给 EP 提供 `get_preferred_hca`(NUMA-aware 传输规划)、`get_active_ranks`、`get_num_synced_ranks`(MIN all_reduce)、`recover_ranks`/`join_group`(弹性)。
- **PG 复用 TE** 做所有数据搬运;**EP 直接用 Device API**(GPU-initiated)绕过 TE CPU 路径,仅借 TE 建链。
- 多 backend 用静态递增 `backendIndex_` 区分(Store key 带 index),P2PDeviceWorker/MooncakeWorker/ConnectionPoller 跨 backend 共享。

### 待后续梳理(TODO)
- [ ] Python 集成层(SGLang/vLLM connector)如何调用 Store/TE
