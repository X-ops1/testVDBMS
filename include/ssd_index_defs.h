#pragma once

#include <filesystem>
#include "utils.h"
#include "utils/tsl/robin_set.h"

#define SECTOR_LEN 4096
#define MAX_N_SECTOR_READS 128
// Both unaligned and aligned.
// example: a record locates in [300, 500], then
// offset = 0, len = 4096 (aligned read for disk)
// u_offset = 300, u_len = 200 (unaligned read)
// Unaligned read: read u_len from u_offset, read to buf + 0.
struct IORequest {
  uint64_t offset;    // where to read from (page)
  uint64_t len;       // how much to read
  void *buf;          // where to read into
  bool finished;      // for async IO
  uint64_t u_offset;  // where to read from (unaligned)
  uint64_t u_len;     // how much to read (unaligned)
  void *base;         // starting address of this sector scratch

  IORequest() : offset(0), len(0), buf(nullptr) {
  }

  IORequest(uint64_t offset, uint64_t len, void *buf, uint64_t u_offset, uint64_t u_len, void *base = nullptr)
      : offset(offset), len(len), buf(buf), u_offset(u_offset), u_len(u_len), base(base) {
    assert(IS_512_ALIGNED(offset));
    assert(IS_512_ALIGNED(len));
    assert(IS_512_ALIGNED(buf));
    // assert(malloc_usable_size(buf) >= len);
  }
};

namespace pipeann {
  template<typename T>
  struct DiskNode {
    uint32_t id = 0;
    T *coords = nullptr;
    uint32_t nnbrs;
    uint32_t *nbrs;

    // id : id of node
    // sector_buf : sector buf containing `id` data
    DiskNode(uint32_t id, T *coords, uint32_t *nhood);
  };

  template<typename T>
  struct QueryBuffer {
    T *coord_scratch = nullptr;  // MUST BE AT LEAST [aligned_dim], for current vector in comparison.

    char *sector_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN], for sectors.
    uint64_t sector_idx = 0;         // index of next [SECTOR_LEN] scratch to use

    float *nbr_ctx_scratch = nullptr;       // MUST BE AT LEAST [256 * NCHUNKS], for pq table distance.
    float *aligned_dist_scratch = nullptr;  // MUST BE AT LEAST pipeann MAX_DEGREE, for exact dist.
    uint8_t *nbr_vec_scratch = nullptr;     // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE], for neighbor PQ vectors.
    T *aligned_query_T = nullptr;
    char *update_buf = nullptr;  // Dynamic allocate in insert_in_place.
    char *onehop_buf = nullptr;

    tsl::robin_set<uint64_t> *visited = nullptr;
    tsl::robin_set<unsigned> *page_visited = nullptr;
    IORequest reqs[MAX_N_SECTOR_READS];

    void reset() {
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
    }
  };

  template<class T>
  struct SSDIndexMetadata {
    // The order matches that on SSD.
    uint32_t nr, nc;
    uint64_t npoints;  // size.
    uint64_t data_dim;
    uint64_t entry_point;
    uint64_t max_node_len;  // without data.
    uint64_t nnodes_per_sector;
    uint64_t npts_cur_shard;
    enum DataType : uint64_t { UNDEFINED = 0, FLOAT = 1, UINT8 = 2, INT8 = 3 } data_type;
    uint64_t max_npts;  // capacity.
    uint64_t range;     // maximum out-degree.

    SSDIndexMetadata() {
    }

    SSDIndexMetadata(uint64_t npoints, uint64_t data_dim, uint64_t entry_point, uint64_t max_node_len,
                     uint64_t nnodes_per_sector)
        : npoints(npoints), data_dim(data_dim), entry_point(entry_point), max_node_len(max_node_len),
          nnodes_per_sector(nnodes_per_sector), npts_cur_shard(npoints), data_type(UNDEFINED) {
      this->max_npts = npoints;
      this->range = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
    }

    void print() const {
      LOG(INFO) << "Max npts: " << max_npts << " Npoints: " << npoints << " Entry point: " << entry_point
                << " Data dim: " << data_dim << " Range: " << range;
      LOG(INFO) << "Max node len: " << max_node_len << " Nnodes per sector: " << nnodes_per_sector
                << " Npts cur shard: " << npts_cur_shard;
    }

    void load_from_disk_index(const std::string &filename, bool sharded = false) {
      if (std::filesystem::exists(filename) == false) {
        LOG(ERROR) << "File " << filename << " does not exist.";
        exit(-1);
      }
      std::ifstream in(filename, std::ios::binary);
      load_from_disk_index(in, sharded);
      in.close();
    }

    void load_from_disk_index(std::ifstream &in, bool sharded = false) {
      LOG(INFO) << "Loading metadata from disk index, sharded: " << sharded;
      in.read((char *) &nr, sizeof(uint32_t));
      in.read((char *) &nc, sizeof(uint32_t));

      in.read((char *) &npoints, sizeof(uint64_t));
      in.read((char *) &data_dim, sizeof(uint64_t));

      in.read((char *) &entry_point, sizeof(uint64_t));
      in.read((char *) &max_node_len, sizeof(uint64_t));
      in.read((char *) &nnodes_per_sector, sizeof(uint64_t));

      this->max_npts = this->npoints;
      if (sharded) {
        in.read((char *) &npts_cur_shard, sizeof(uint64_t));
      } else {
        npts_cur_shard = npoints;
      }

      this->range = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
    }

    void save_to_disk_index(const std::string &filename) {
      std::ofstream out(filename, std::ios::in | std::ios::out | std::ios::binary);
      save_to_disk_index(out);
      out.close();
    }

    void save_to_disk_index(std::ofstream &out) {
      nr = 6;  // hard-coded for the number of uint64_t below.
      nc = 1;
      out.write((char *) &nr, sizeof(uint32_t));
      out.write((char *) &nc, sizeof(uint32_t));

      out.write((char *) &npoints, sizeof(uint64_t));
      out.write((char *) &data_dim, sizeof(uint64_t));

      out.write((char *) &entry_point, sizeof(uint64_t));
      out.write((char *) &max_node_len, sizeof(uint64_t));
      out.write((char *) &nnodes_per_sector, sizeof(uint64_t));
      out.write((char *) &npts_cur_shard, sizeof(uint64_t));
    }
  };
};  // namespace pipeann

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif
