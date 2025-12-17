#include "neighbor.h"
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "v2/dynamic_index.h"
#include <csignal>
#include <cstdint>
#include <mutex>
#include <vector>

#include <algorithm>
#include <filesystem>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <shared_mutex>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <gperftools/malloc_extension.h>

#include "aux_utils.h"
#include "ssd_index.h"

#include "linux_aligned_file_reader.h"

#include "v2/l1_neighbor_table.h"

namespace pipeann {

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::wolverine_delete( const TagT &tag) {
    // mark as deleted
    // {
    // std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    // if (deletion_sets[active_delete_set].find(tag) == deletion_sets[active_delete_set].end()) {
    //   deletion_sets[active_delete_set].insert(tag);
    // }
    // }
    this->_disk_index->delete_in_place( tag, &deletion_sets[active_delete_set]);
    // {
    // std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    //   deletion_sets[active_delete_set].erase(tag);
    // }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::delete_in_place( const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    uint32_t del_id = this->tag2id( tag);
    // ssd: vec_lock_table
    // ssd: page_lock_table
    // memory: idx_lock_table
    // memory: page_idx_lock_table

    uint32_t del_loc = optimize_lock_id( del_id, true);
    uint64_t del_page_no = loc_sector_no( del_loc);

    auto read_data = pop_query_buf(nullptr);
    void *ctx = reader->get_ctx();
    
    std::vector<IORequest> del_page_read_req;
    char *del_page_buf = read_data->sector_scratch;
    auto *del_node_buf = offset_to_loc( del_page_buf, del_loc);
    T *del_vec = offset_to_node_coords( del_page_buf);
    // *del_nbr is # nbrs, nbr ids begin from *(del_nbr+1)
    unsigned *del_nbr = offset_to_node_nhood( del_node_buf);
    del_page_read_req.emplace_back( del_page_no * SECTOR_LEN, size_per_io, del_page_buf, 0, 0);
    reader->read_alloc(del_page_read_req, ctx);
    page_lock_table.unlock( del_page_no);

    std::vector<uint64_t> page2deref;
    page2deref.push_back( del_page_no);
    reader->deref( &page2deref, ctx);

    unsigned del_nnbrs = *del_nbr;
    del_nbr += 1;
    std::vector<IORequest> nxt_read_req;
    assert(read_data->onehop_buf == nullptr);
    pipeann::alloc_aligned((void **) &read_data->onehop_buf, this->range * size_per_io, SECTOR_LEN);
    auto &onehop_buf = read_data->onehop_buf;
    
    std::unordered_map<unsigned, int> hop12_nbr_map;
    std::vector<unsigned> hop12_nbr_array;
    hop12_nbr_array.reserve( this->range * (this->range + 1));
    unsigned valid_nnbrs = 0;

    std::unordered_map<uint32_t, char*> page2buf;
    std::vector<uint32_t> snapshot_loc;
    snapshot_loc.reserve( this->range);
    uint64_t page_enum_id = 0;

    std::vector<bool> loaded( del_nnbrs, false);

    auto collect_one_hop = [&]( unsigned ii, uint32_t loc) {
      loaded[ii] = true;
      snapshot_loc.push_back( loc);
      valid_nnbrs++;
      hop12_nbr_map[ del_nbr[ii]] = hop12_nbr_array.size();
      hop12_nbr_array.push_back( del_nbr[ii]);
    };

    for( unsigned i = 0; i < del_nnbrs; i++) {
      if( loaded[i]) {
        continue;
      }
      nxt_read_req.clear();
      uint32_t cur_loc = optimize_lock_id( del_nbr[i], true);
      uint64_t cur_page = loc_sector_no( cur_loc);
      if( cur_loc == kInvalidID || deletion_set->find(del_nbr[i]) != deletion_set->end()) {
        page_lock_table.unlock( cur_page);
        continue;
      }
      page2buf[cur_page] = onehop_buf + page_enum_id * size_per_io;
      collect_one_hop( i, cur_loc);
      PageArr layout = get_page_layout(cur_page);
      for( auto &vertex : layout) {
        if( vertex == del_nbr[i] || vertex == kInvalidID) {
          continue;
        }
        // idx_lock_table.rdlock( vertex);
        if( id2loc( vertex) == kInvalidID || deletion_set->find(vertex) != deletion_set->end()) {
          continue;
        }
        auto iter = std::find( del_nbr, del_nbr + del_nnbrs, vertex);
        if( iter == del_nbr + del_nnbrs) {
          continue;
        }
        collect_one_hop( (unsigned)( iter - del_nbr), id2loc( vertex));
        // idx_lock_table.unlock( vertex);
      }
      nxt_read_req.emplace_back( cur_page * SECTOR_LEN, size_per_io, onehop_buf + page_enum_id * size_per_io, 0, 0);
      reader->read_alloc(nxt_read_req, ctx);
      page2deref.clear();
      page2deref.push_back( cur_page);
      reader->deref( &page2deref, ctx);
      page_enum_id++;
      page_lock_table.unlock( cur_page);
    }

    for( unsigned i = 0; i < valid_nnbrs; i++) {
      uint32_t cur_loc = snapshot_loc[i];
      uint64_t cur_page = loc_sector_no( cur_loc);
      auto *cur_nbr_buf = offset_to_loc( page2buf[cur_page], cur_loc);
      unsigned *cur_nbr_nbr = offset_to_node_nhood( cur_nbr_buf);
      unsigned cur_nbr_nnbrs = *cur_nbr_nbr;
      assert( cur_nbr_nnbrs > 0);
      cur_nbr_nbr += 1;
      for( unsigned j = 0; j < cur_nbr_nnbrs; j++) {
        idx_lock_table.rdlock( cur_nbr_nbr[j]);
        assert( cur_nbr_nbr[j] != kInvalidID);
        assert( cur_nbr_nbr[j] != kAllocatedID);
        if( id2loc( cur_nbr_nbr[j]) != kInvalidID && deletion_set->find(cur_nbr_nbr[j]) == deletion_set->end() &&
            hop12_nbr_map.find( cur_nbr_nbr[j]) == hop12_nbr_map.end()) {
          hop12_nbr_map[ cur_nbr_nbr[j]] = hop12_nbr_array.size();
          hop12_nbr_array.push_back( cur_nbr_nbr[j]);
        }
        idx_lock_table.unlock( cur_nbr_nbr[j]);
      }
    }

    unsigned npts_hop12 = hop12_nbr_array.size();
    std::vector<unsigned> cnt_bu( valid_nnbrs, 0);
    // LOG(INFO) << "# 1hop: " << valid_nnbrs << "; # 2hop: " << npts_hop12;
    std::vector<float> dists_del2hop12_SDC( npts_hop12, 0.0f);
    std::vector<float> dists_del2hop12_ADC( npts_hop12, 0.0f);
    auto &thread_pq_buf = read_data->nbr_vec_scratch;
    nbr_handler->compute_dists( del_id, hop12_nbr_array.data(), npts_hop12, 
                                dists_del2hop12_SDC.data(), thread_pq_buf);
    auto tmp_qb = pop_query_buf( del_vec);
    aligned_free( tmp_qb->aligned_dist_scratch);
    pipeann::alloc_aligned((void **) &tmp_qb->aligned_dist_scratch, 4096 * sizeof(float), 256);
    float *dist_scratch = tmp_qb->aligned_dist_scratch;
    nbr_handler->compute_dists( tmp_qb, hop12_nbr_array.data(), npts_hop12);
    for( unsigned i = 0; i < npts_hop12; i++) {
      dists_del2hop12_ADC[i] = dist_scratch[i];
    }
    std::vector<std::vector<unsigned>> id_edge_from( npts_hop12);
    for( unsigned i = 0; i < valid_nnbrs; i++) {
      std::vector<float> dists_nbr2hop12_SDC( npts_hop12, 0.0f);
      nbr_handler->compute_dists( hop12_nbr_array[i], hop12_nbr_array.data(), 
                                  npts_hop12, dists_nbr2hop12_SDC.data(), thread_pq_buf);
      float d_p_pout_sdc = dists_del2hop12_SDC[i];
      float d_p_pout_adc = dists_del2hop12_ADC[i];
      for( unsigned j = 0; j < npts_hop12; j++) {
        if( i == j) continue;
        // squared dists
        float d_x_p_sdc = dists_del2hop12_SDC[j];
        float d_x_p_adc = dists_del2hop12_ADC[j];
        float d_x_pout_sdc = dists_nbr2hop12_SDC[j];
        if( d_x_p_adc > d_p_pout_adc + 1e-7 && 2.0f * d_p_pout_adc > d_x_p_adc + 1e-7 &&
            d_x_pout_sdc < d_p_pout_sdc - 1e-7 && d_x_pout_sdc + d_p_pout_sdc > d_x_p_sdc + 1e-7) {
          id_edge_from[j].push_back( hop12_nbr_array[i]);
          cnt_bu[i]++;
          // if( cnt_bu[i] > this->range) break;
        }
      }
    }
    this->push_query_buf(tmp_qb);

    std::vector<bool> modified( npts_hop12, false);

    for( unsigned i = 0; i < npts_hop12; i++) {
      if( modified[i] || id_edge_from[i].empty()) {
        continue;
      }
      uint32_t cur_loc = optimize_lock_id( hop12_nbr_array[i], false);
      uint64_t cur_page = loc_sector_no( cur_loc);
      if( cur_loc == kInvalidID || deletion_set->find(hop12_nbr_array[i]) != deletion_set->end()) {
        page_lock_table.unlock( cur_page);
        continue;
      }

      std::vector<std::pair<uint32_t, uint32_t>> page_append;
      std::vector<uint32_t> id2check;
      PageArr layout = get_page_layout(cur_page);
      for( auto &cur_id : layout) {
        if( cur_id == kInvalidID) {
          continue;
        }
        id2check.push_back( cur_id);
        if( hop12_nbr_map.find( cur_id) == hop12_nbr_map.end()) {
          continue;
        }
        int tmp = hop12_nbr_map[cur_id];
        assert( modified[tmp] == false);
        modified[tmp] = true;
        if( id_edge_from[tmp].empty()) {
          continue;
        }
        if( id2loc(cur_id) == kInvalidID || deletion_set->find(cur_id) != deletion_set->end()) {
          continue;
        }
        // id2check.push_back( cur_id);
        for( auto &nbr : id_edge_from[tmp]) {
          page_append.emplace_back( cur_id, nbr);
        }
      }
      for( unsigned j = 0; j < page_append.size(); j++) {
        bool check_merge = (j==page_append.size()-1) ? true : false;
        l1_table_->add_backlink( page_append[j].first, page_append[j].second, 
                                 deletion_set, check_merge, &id2check);
      }
      page_lock_table.unlock( cur_page);
    }
    aligned_free( read_data->onehop_buf);
    read_data->onehop_buf = nullptr;
    del_loc = optimize_lock_id( del_id, true);
    del_page_no = loc_sector_no( del_loc);
    idx_lock_table.wrlock( del_id);
    set_id2loc( del_id, kInvalidID);
    set_loc2id( del_loc, kInvalidID);
    idx_lock_table.unlock( del_id);
    page_lock_table.unlock( del_page_no);
    this->push_query_buf(read_data);
    // deletion_set->erase( tag);
  }

  template class DynamicSSDIndex<float>;
  template class DynamicSSDIndex<uint8_t>;
  template class DynamicSSDIndex<int8_t>;
}