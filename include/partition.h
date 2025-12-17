#pragma once

#include <cassert>
#include <string>
#include <memory>
#include <vector>
#include "partition_tree.h"

const uint32_t NUM_PQ_CENTERS = 256;
const uint32_t NUM_K_MEANS_ITERS = 15;

template<typename T>
void gen_random_slice(const std::string base_file, const std::string output_prefix, double sampling_rate,
                      size_t offset = 0);

template<typename T>
void gen_random_slice(const std::string data_file, double p_val, std::unique_ptr<float[]> &sampled_data,
                      size_t &slice_size, size_t &ndims);

template<typename T>
void gen_random_slice(const std::string data_file, double p_val, float *&sampled_data, size_t &slice_size,
                      size_t &ndims);

template<typename T>
void gen_random_slice(const T *inputdata, size_t npts, size_t ndims, double p_val, float *&sampled_data,
                      size_t &slice_size);

template<typename T>
int estimate_cluster_sizes(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                           const size_t k_base, std::vector<size_t> &cluster_sizes);

template<typename T>
int shard_data_into_clusters(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                             const size_t k_base, std::string prefix_path);

template<typename T>
int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                              size_t graph_degree, const std::string prefix_path, size_t k_base);

template<typename T>
float split_into_two_clusters(float *data, float *pivots, const size_t num, const size_t dim,
                              std::vector<size_t> &cluster_sizes, std::vector<float *> &cluster_data, int level,
                              int nid);

template<typename T>
void create_partition_tree(float *data, size_t num, size_t dim, const double sampling_rate, double budget,
                           size_t graph_degree, partitionNode *root);

template<typename T>
int shard_data_into_clusters_with_tree(const std::string data_file, const std::string idmap_file, partitionNode *root,
                                       std::string prefix_path);

template<typename T>
int hierarchical_partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                                           size_t graph_degree, const std::string prefix_path);
