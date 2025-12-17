#pragma once

#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

#include <unistd.h>

#include "nbr/abstract_nbr.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"

namespace pipeann {
    template<typename T, typename TagT>
    class SSDIndex;

    double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                            unsigned *our_results, unsigned dim_or, unsigned recall_at);

    double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                            unsigned *our_results, unsigned dim_or, unsigned recall_at,
                            const tsl::robin_set<unsigned> &active_tags);

    void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs);

    int
    merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                 const std::string &idmaps_suffix, const uint64_t nshards, unsigned max_degree,
                 const std::string &output_vamana, const std::string &medoids_file,
                 const std::string &index_prefix_path = "");

    template<typename T>
    int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, bool single_index_file,
                                  unsigned L, unsigned R, double sampling_rate, double ram_budget,
                                  std::string mem_index_path, std::string medoids_file, std::string centroids_file,
                                  const char *tag_file = nullptr);

    template<typename T>
    int build_merged_vamana_index_my(std::string base_file, pipeann::Metric _compareMetric, unsigned L, unsigned R,
                                     double sampling_rate, double ram_budget, std::string mem_index_path,
                                     std::string medoids_file, std::string index_prefix_path = "");

    template<typename T, typename TagT = uint32_t>
    bool build_disk_index(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                          uint32_t num_threads, uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                          const char *tag_file, AbstractNeighbor<T> *nbr_handler);

    template<typename T, typename TagT = uint32_t>
    void
    create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                       const std::string &output_file);
}  // namespace pipeann
