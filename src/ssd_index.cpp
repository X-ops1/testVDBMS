#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>
#include <filesystem>

#include <omp.h>
#include <cmath>
#include "nbr/abstract_nbr.h"
#include "ssd_index_defs.h"
#include "utils/timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "utils/tsl/robin_set.h"

#include <algorithm>
#include <map>

namespace pipeann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader,
                              AbstractNeighbor<T> *nbr_handler, bool tags, Parameters *params)
      : reader(fileReader), nbr_handler(nbr_handler), data_is_normalized(false), enable_tags(tags) {
    if (m == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = pipeann::Metric::L2;
        data_is_normalized = true;

      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " This may result in erroneous results or poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

    if (params != nullptr) {
      this->beam_width = params->beam_width;
      this->l_index = params->L;
      this->range = params->R;
      this->maxc = params->C;
      this->alpha = params->alpha;
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
    LOG(INFO) << "Use " << nbr_handler->get_name() << " as neighbor handler.";
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << v2::cache.cache.size();

    if (load_flag) {
      this->destroy_buffers();
      reader->close();
    }
  }

  // --- merge 队列的简单封装 ---
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::push(uint32_t id) {
    // 简单过滤不把空值推到队列里
    if (id == MERGE_NULL_ID) {
      return;
    }
    merge_queue_.push(id);
    // 唤醒在 pop_blocking 中等待的线程
    merge_queue_.push_notify_all();
  }

  template<typename T, typename TagT>
  bool SSDIndex<T, TagT>::try_pop(uint32_t &id) {
    // ConcurrentQueue 的 pop 是非阻塞的，队列空时会返回 null_T
    id = merge_queue_.pop();
    if (id == merge_queue_.null_T) {
      return false;
    }
    return true;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::pop_blocking(uint32_t &id) {
    // 先尝试一次 pop，如果是空值，就进入等待
    id = merge_queue_.pop();
    while (id == merge_queue_.null_T) {
      // 没有新的任务，等待 push 通知
      merge_queue_.wait_for_push_notify();
      id = merge_queue_.pop();
    }
  }


  // --- L1 增量图合并线程：从队列取节点，按扇区分组，然后调用 merge_nodes_on_sector ---
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_worker_thread() {
    LOG(INFO) << "SSDIndex merge worker thread started.";

    // 为当前线程创建 I/O 上下文（io_uring / AIO）
    void *ctx = reader->get_ctx();

    // 单次批处理最多从队列拉这么多点，可以后面再调参
    constexpr size_t kMergeBatchSize = 10000;

    std::vector<uint32_t> batch;
    batch.reserve(kMergeBatchSize);

    auto timer = pipeann::Timer();

    while (true) {
      uint32_t id = 0;

      // 1）阻塞式拿到至少一个节点 id
      pop_blocking(id);

      // 收到终止哨兵，直接退出
      if (id == MERGE_TERMINATE_ID) {
        LOG(INFO) << "Merge worker thread received terminate sentinel.";
        break;
      }

      // 简单过滤非法 id
      if (id < num_points) {
        batch.push_back(id);
        // 统计：merge 线程从队列里成功取出一个合法 id
        stats_merge_pop_.fetch_add(1);
      }

      // 2）非阻塞地尽可能多地把当前队列里的任务拉出来，形成一个 batch
      uint32_t tmp = 0;
      while (batch.size() < kMergeBatchSize && try_pop(tmp)) {
        if (tmp == MERGE_TERMINATE_ID) {
          // 如果还有任务没处理完，就把终止哨兵丢回去，留给下一轮退出
          push(MERGE_TERMINATE_ID);
          break;
        }
        if (tmp < num_points) {
          batch.push_back(tmp);
          // 同样这里也统计一次 pop 成功
          stats_merge_pop_.fetch_add(1);
        }
      }

      if (batch.empty()) {
        // 有可能刚才只拿到了终止哨兵或非法 id，继续下一轮
        continue;
      }
      // if(timer.elapsed() >= 5000000)
      // {
        LOG(INFO) << "[MERGE] new batch: raw_ids=" << batch.size();
      // }

      // 去重前的 batch 大小，可以作为一个观察值
      const size_t before_dedup = batch.size();

      // 3）去重：同一个点在队列里出现多次，只合并一次
      std::sort(batch.begin(), batch.end());
      batch.erase(std::unique(batch.begin(), batch.end()), batch.end());

      const size_t after_dedup = batch.size();
      // 累加这次真正要 merge 的节点数
      auto merged_total = stats_merge_nodes_merged_.fetch_add(after_dedup) + after_dedup;

      // if(timer.elapsed() >= 5000000)
      // {
        LOG(INFO) << "[MERGE] new batch true merge: raw_ids=" << after_dedup;
      // }

      // 每 1000 个节点左右打印一次大致统计信息
      if (merged_total % 1000 == 0) {
        LOG(INFO) << "[MERGE] merged_nodes=" << merged_total
                  << ", l1_triggers=" << stats_l1_merge_triggers_.load()
                  << ", popped_ids=" << stats_merge_pop_.load()
                  << ", last_batch_before_dedup=" << before_dedup
                  << ", last_batch_after_dedup=" << after_dedup;
      }

      // 4）按扇区号分组：sector -> [nodes...]
      std::map<uint64_t, std::vector<uint32_t>> sector_nodes;
      for (uint32_t vid : batch) {
        uint64_t sector = node_sector_no(vid);  // 利用已有的 id -> loc -> sector 映射
        sector_nodes[sector].push_back(vid);
      }

      // if(timer.elapsed() >= 5000000)
      // {
        LOG(INFO) << "[MERGE] batch sectors=" << sector_nodes.size();
        timer.reset();
      // }

      // 5）对每个扇区调用真正的合并逻辑
      //   读盘 + 抽干 L1 增量邻居 + pool 合并 + prune_neighbors_pq + BgTask + wbc_write
      for (auto &kv : sector_nodes) {
        uint64_t sector = kv.first;
        const std::vector<uint32_t> &nodes = kv.second;
        merge_nodes_on_sector(sector, nodes, ctx);
      }

      // 6）清空 batch，处理下一轮
      batch.clear();
    }

    LOG(INFO) << "SSDIndex merge worker thread exited. "
              << "l1_triggers=" << stats_l1_merge_triggers_.load()
              << ", popped_ids=" << stats_merge_pop_.load()
              << ", merged_nodes=" << stats_merge_nodes_merged_.load();
  }

  // 新方案入口：插入线程内的“整页搬迁 + 合并”。
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_page_for_node_inline(uint32_t center) {
  #if SSD_L1_MERGE_MODE == SSD_L1_MERGE_INLINE_PAGE
    // 0. 简单过滤非法 id
    if (center >= this->num_points) {
      return;
    }

    // 1. 通过 id -> page 号
    uint64_t page = this->id2page(center);
    if (page == kInvalidID) {
      LOG(WARNING) << "[MERGE][INLINE] center=" << center
                  << " has invalid page, skip.";
      return;
    }

    // 2. 拿到这一页的布局：page 上每个 slot 当前对应的 id
    //    get_page_layout 会返回该页所有 loc 上的 id
    auto layout = this->get_page_layout(static_cast<uint32_t>(page));

    std::vector<uint32_t> nodes_in_page;
    nodes_in_page.reserve(layout.size());
    for (uint32_t vid : layout) {
      // 过滤掉无效 / 仅占位的 slot
      if (vid == kInvalidID || vid == kAllocatedID) {
        continue;
      }
      nodes_in_page.push_back(vid);
    }

    if (nodes_in_page.empty()) {
      LOG(WARNING) << "[MERGE][INLINE] page=" << page
                  << " has no valid nodes, center=" << center;
      return;
    }

    // 3. 打一点调试日志
    static thread_local uint64_t local_cnt = 0;
    ++local_cnt;
    if ((local_cnt & ((1u << 12) - 1)) == 0) {  // 每 4096 次触发打一条
      LOG(INFO) << "[MERGE][INLINE] center=" << center
                << " page="   << page
                << " nodes_in_page=" << nodes_in_page.size();
    }

    // 4. 复用原有的按扇区 merge 逻辑：
    //    L0[v] + L1[v] → cand → PQ 剪枝 → 新的 L0[v]，并写回这一页
    void *ctx = reader->get_ctx();
    this->merge_nodes_on_sector(page, nodes_in_page, ctx);
  #else
    (void) center;
  #endif
  }




  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::copy_index(const std::string &prefix_in, const std::string &prefix_out) {
    LOG(INFO) << "Copying disk index from " << prefix_in << " to " << prefix_out;
    std::filesystem::copy(prefix_in + "_disk.index", prefix_out + "_disk.index",
                          std::filesystem::copy_options::overwrite_existing);
    if (std::filesystem::exists(prefix_in + "_disk.index.tags")) {
      std::filesystem::copy(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags",
                            std::filesystem::copy_options::overwrite_existing);
    } else {
      // remove the original tags.
      std::filesystem::remove(prefix_out + "_disk.index.tags");
    }

    // nbr.
    this->nbr_handler->load(prefix_in.c_str());
    this->nbr_handler->save(prefix_out.c_str());

    // partition data
    if (std::filesystem::exists(prefix_in + "_partition.bin.aligned")) {
      std::filesystem::copy(prefix_in + "_partition.bin.aligned", prefix_out + "_partition.bin.aligned",
                            std::filesystem::copy_options::overwrite_existing);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(uint64_t n_threads) {
    uint64_t n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    this->thread_data_queue.null_T = nullptr;
    this->merge_queue_.null_T = MERGE_NULL_ID;
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
      this->reader->register_buf(data->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
    }

#ifndef READ_ONLY_TESTS
    // background thread: 负责 BgTask 的写盘
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }

    // 根据 SSD_L1_MERGE_MODE 决定是否启用后台 L1 merge 线程
#if SSD_L1_MERGE_MODE == SSD_L1_MERGE_BG_WORKER
    LOG(INFO) << "Setup L1 merge worker thread for disk-graph merging (BG_WORKER mode)...";
    std::thread(&SSDIndex<T, TagT>::merge_worker_thread, this).detach();
#else
    LOG(INFO) << "L1 merge worker thread disabled (INLINE_PAGE mode, merge in insert path).";
#endif

#endif
}



  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_buffers() {
#ifndef READ_ONLY_TESTS
    for (int i = 0; i < kBgIOThreads; ++i) {
      if (bg_io_thread_[i] != nullptr) {
        auto bg_task = new BgTask{
            .thread_data = nullptr, .writes = {}, .pages_to_unlock = {}, .pages_to_deref = {}, .terminate = true};
        bg_tasks.push(bg_task);
        bg_tasks.push_notify_all();
        bg_io_thread_[i] = nullptr;
      }
    }
#endif

    while (!this->thread_data_bufs.empty()) {
      auto buf = this->thread_data_bufs.back();
      pipeann::aligned_free((void *) buf->coord_scratch);
      pipeann::aligned_free((void *) buf->sector_scratch);
      pipeann::aligned_free((void *) buf->nbr_vec_scratch);
      pipeann::aligned_free((void *) buf->nbr_ctx_scratch);
      pipeann::aligned_free((void *) buf->aligned_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_query_T);
      this->thread_data_bufs.pop_back();
      this->thread_data_queue.pop();
      delete buf;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path) {
    if (mem_index_path.empty()) {
      LOG(ERROR) << "mem_index_path is needed";
      exit(-1);
    }
    mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(metric, query_dim, 0, false, false, true);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, uint32_t num_threads, bool new_index_format,
                              bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
    this->_disk_index_file = disk_index_file;

    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  uint64_t SSDIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector,
                                           const uint64_t num_points) {
    // 完全忽略 partition 文件，使用等距映射。
    // 这里也干脆去掉 OpenMP 并行，反正只初始化一次，3M 点也就几十毫秒级。
    (void) index_prefix;      // 不再使用 partition.bin
    (void) nnodes_per_sector; // 不再依赖 C
    (void) num_points;        // 使用 this->num_points 更稳妥

    // 保险起见，让 num_points / cur_loc 完全由 metadata 决定
    const size_t npts   = static_cast<size_t>(this->num_points);
    const size_t nslots = static_cast<size_t>(this->cur_loc);

    id2loc_.assign(npts, 0u);           // 大小 = npts
    loc2id_.assign(nslots, kInvalidID); // 大小 = cur_loc

    // 0..(npts-1)：等距映射 id -> loc
    for (size_t i = 0; i < npts; ++i) {
      id2loc_[i] = static_cast<uint32_t>(i);
      loc2id_[i] = static_cast<uint32_t>(i);
    }

    // npts..(nslots-1)：是最后一页对齐产生的 padding，统一标为无效
    // 上面 assign 已经全部设成 kInvalidID 了，这里实际上可以不写。
    LOG(INFO) << "Page layout loaded with equal mapping (no per-page reserved slots). "
              << "Npoints=" << npts << " Cur_loc=" << nslots;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
      LOG(INFO) << "Loaded " << tags.size() << " tags";
    }
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::get_vector_by_id(const uint32_t &id, T *vector_coords) {
    if (!enable_tags) {
      LOG(INFO) << "Tags are disabled, cannot retrieve vector";
      return -1;
    }
    uint32_t pos = id;
    size_t num_sectors = node_sector_no(pos);
    std::ifstream disk_reader(_disk_index_file.c_str(), std::ios::binary);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
    disk_reader.seekg(SECTOR_LEN * num_sectors, std::ios::beg);
    disk_reader.read(sector_buf.get(), size_per_io);
    char *node_coords = (offset_to_node(sector_buf.get(), pos));
    memcpy((void *) vector_coords, (void *) node_coords, data_dim * sizeof(T));
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
