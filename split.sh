#!/usr/bin/env bash
set -euo pipefail

ROOT="/NV1/czy/testVDBMS/data&index/BIGANN_1M"
SRC_DATA="/DATA/dataset/SIFT/bigann/bigann_learn.bin"
QUERY_SRC="/DATA/dataset/SIFT/bigann/bigann_query.bin"

# mkdir -p "$ROOT/gt" "$ROOT/logs"

# # 1. 采样base和插入（切换到目标目录采样，确保输出文件在$ROOT下）
# cd "$ROOT"
# # 采样 1M（概率 0.01）
# /NV1/czy/testVDBMS/build/tests/utils/gen_random_slice uint8 "$SRC_DATA" base_1M 0.01
# # 采样 4M（概率 0.04）
# /NV1/czy/testVDBMS/build/tests/utils/gen_random_slice uint8 "$SRC_DATA" insert_4M 0.04
# cd -

# # 2. 拼接base+insert
# python3 -c "
# import numpy as np
# def read_fbin(path):
#     with open(path, 'rb') as f:
#         n, d = np.fromfile(f, dtype=np.int32, count=2)
#         data = np.fromfile(f, dtype=np.uint8).reshape(n, d)
#     return data, d
# base, dim = read_fbin('$ROOT/base_1M_data.bin')
# insert, _ = read_fbin('$ROOT/insert_4M_data.bin')
# all_data = np.vstack([base, insert])
# with open('$ROOT/data_1M_4M.bin', 'wb') as f:
#     np.array([all_data.shape[0], dim], dtype=np.int32).tofile(f)
#     all_data.astype(np.uint8).tofile(f)
# "

# # 3. 拷贝查询向量
# cp "$QUERY_SRC" "$ROOT/bigann_query.bin"

# # 4. 构建索引
# /NV1/czy/testVDBMS/build/tests/build_disk_index uint8 "$ROOT/base_1M_data.bin" "$ROOT/index_1M" 96 128 32 256 112 l2 pq

# # 5. 生成全量ground-truth
# /NV1/czy/testVDBMS/build/tests/utils/compute_groundtruth uint8 "$ROOT/data_1M_4M.bin" "$ROOT/bigann_query.bin" 1000 "$ROOT/full_gt.bin"

# # 6. 切分ground-truth
# /NV1/czy/testVDBMS/build/tests/gt_update "$ROOT/full_gt.bin" 1000516 5000689 100000 10 "$ROOT/gt" 1

# 7. 运行插入+查询实验
/NV1/czy/testVDBMS/build/tests/test_insert_search uint8 \
    "$ROOT/data_1M_4M.bin" 128 100000 10 10 32 0 "$ROOT/index_1M" "$ROOT/bigann_query.bin" "$ROOT/gt" 0 10 4 4 0 20 | tee "$ROOT/logs/odiann_hello.log"

# 8. 清理shadow索引
rm -f "$ROOT/index_1M"_shadow*