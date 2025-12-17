//
// Created by owlet on 2025/11/19.
//

#ifndef PIPEANN_PARTITION_TREE_H
#define PIPEANN_PARTITION_TREE_H

#include <iostream>
#include <vector>
#include "index.h"

class partitionNode {
    struct leafNode {
        size_t cluster_size;

        leafNode(size_t sz) : cluster_size(sz) {
        }
    };

    struct internalNode {
        float boundary_factor;
        std::vector<float> centroids;
        std::unique_ptr<partitionNode> left;
        std::unique_ptr<partitionNode> right;

        internalNode(float bf, std::vector<float> &pivots) : boundary_factor(bf) {
            centroids = std::move(pivots);
        }
    };

private:
    bool is_leaf = false;
    int level;
    int nid;
    std::unique_ptr<leafNode> leaf;
    std::unique_ptr<internalNode> internal;

public:
    partitionNode(bool is_leaf, int level, int id) : is_leaf(is_leaf), level(level), nid(id) {
    }

    void make_leaf(size_t sz) {
        assert(is_leaf);
        leaf = std::make_unique<leafNode>(sz);
    }

    void make_internal(float bf, std::vector<float> &pivots,
                       std::unique_ptr<partitionNode> left_node, std::unique_ptr<partitionNode> right_node) {
        assert(!is_leaf);
        internal = std::make_unique<internalNode>(bf, pivots);
        internal->left = std::move(left_node);
        internal->right = std::move(right_node);
    }

    void rename_shard(const std::string prefix_path, int &shard_id) {
        if (is_leaf) {
            std::string old_data_file =
                    prefix_path + "_level-" + std::to_string(level) + "_subshard-" + std::to_string(nid) + ".bin";
            std::string old_idmap_file =
                    prefix_path + "_level-" + std::to_string(level) + "_subshard-" + std::to_string(nid) +
                    "_ids_uint32.bin";
            std::string new_data_file = prefix_path + "_subshard-" + std::to_string(shard_id) + ".bin";
            std::string new_idmap_file = prefix_path + "_subshard-" + std::to_string(shard_id) + "_ids_uint32.bin";
            std::rename(old_data_file.c_str(), new_data_file.c_str());
            std::rename(old_idmap_file.c_str(), new_idmap_file.c_str());
            shard_id++;
            return;
        }
        internal->left->rename_shard(prefix_path, shard_id);
        internal->right->rename_shard(prefix_path, shard_id);
    }

    int build_graph(const std::string prefix_path) {
        if (is_leaf) {
            std::string shard_base_file =
                    prefix_path + "_level-" + std::to_string(level) + "_subshard-" + std::to_string(nid) + ".bin";
            std::string shard_index_file =
                    prefix_path + "_level-" + std::to_string(level) + "_subshard-" + std::to_string(nid) + "_mem.index";

            pipeann::Metric _compareMetric = pipeann::Metric::L2;
            pipeann::Parameters paras;
            paras.set(64, 100, 750, 1.5, 0, false);
            uint64_t shard_base_dim, shard_base_pts;
            pipeann::get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
            std::unique_ptr<pipeann::Index<float>> _pvamanaIndex = std::unique_ptr<pipeann::Index<float>>(
                    new pipeann::Index<float>(_compareMetric, shard_base_dim, shard_base_pts, false, false));
            _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras);
            _pvamanaIndex->save(shard_index_file.c_str());

            return 1;
        }
        int ll = internal->left->build_graph(prefix_path);
        int lr = internal->right->build_graph(prefix_path);

        return std::max(ll, lr) + 1;
    }

    bool check_leaf() {
        return is_leaf;
    }

    int get_level() {
        return level;
    }

    int get_id() {
        return nid;
    }

    leafNode *get_leaf() {
        assert(is_leaf);
        return leaf.get();
    }

    internalNode *get_internal() {
        assert(!is_leaf);
        return internal.get();
    }
};

#endif  // PIPEANN_PARTITION_TREE_H
