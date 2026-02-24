import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch as GraphBatch
from typing import Callable

class BatchClipper():
    '''
    A simple class to clip batches and recycle the clipped batches back into subsequent batches
    
    Args:
        max_nodes (int): Maximum number of nodes allowed in a batch.
        max_cache_samples (int): Maximum number of samples allowed in cache before trimming.
        
        clip_method (str = 'random'): Method to use for clipping samples.
            'random': randomly clip samples until under max nodes
            'largest' : clip largest samples first until under max nodes.
            'smallest' : clip smallest samples first until under max nodes.
        
        pull_method (str = 'largest'): Method to use for pulling samples from cache.
            'random' : randomly pull samples from cache until num_nodes_needed is met
            'largest' : pull largest samples from cache first until num_nodes_needed is met
            'smallest' : pull smallest samples from cache first until num_nodes_needed is met
        
        sample_size_fn (Callable): Function to compute the size of a data sample.
        batch_size_fn (Callable): Function to compute the size of a batch.
        verbose (bool): Whether to print verbose output.
    
    '''
    def __init__(self, max_nodes : int = 2000, 
                        max_cache_samples : int = 1000, 
                        clip_method : str = 'random',
                        pull_method : str = 'largest',
                        sample_size_fn : Callable = None,
                        batch_size_fn: Callable = None,
                        verbose : bool =False
                        ):
        self.max_nodes = max_nodes
        self.max_cache_samples = max_cache_samples
        self.clip_method = clip_method
        self.pull_method = pull_method
        self.cache = []
        self.cache_sizes = []
        self.sample_size_fn = sample_size_fn
        self.batch_size_fn = batch_size_fn
        self.verbose = verbose

        if self.sample_size_fn is None:
            self.sample_size_fn = lambda data: data.num_nodes # default
        
        if self.batch_size_fn is None:
            self.batch_size_fn = lambda batch: batch.num_nodes # default
        
        self.reset()

        self.active = True
    
    def track(self, num_nodes, num_clipped=0, num_added=0):
        #track statistics for the given batch
        self._total_passes += 1
        self._num_nodes += num_nodes
        self._num_nodes_2 += num_nodes**2
        self._total_clipped += num_clipped
        self._total_added += num_added

        if num_nodes > self._max_num_nodes:
            self._max_num_nodes = num_nodes
        if num_nodes < self._min_num_nodes:
            self._min_num_nodes = num_nodes

    def get_stats_dict(self) -> dict:
        stats = {
            "total_passes": self._total_passes,
            "avg_nodes_per_batch": 0.0,
            "std_nodes_per_batch": 0.0,
            "max_nodes_in_batch": self._max_num_nodes,
            "min_nodes_in_batch": self._min_num_nodes,
            "total_clipped_samples": self._total_clipped,
            "total_added_from_cache": self._total_added,
            "total_cache_trim_events": self._trimmed_cache_events,
            "total_samples_trimmed_from_cache": self._trimmed_cache_samples,
            "current_cache_size": len(self.cache),
            "current_cache_nodes": sum(self.cache_sizes),
        }
        if self._total_passes == 0:
            return stats
        
        avg_nodes = self._num_nodes / self._total_passes
        avg_nodes_2 = self._num_nodes_2 / self._total_passes
        std_nodes = (np.sqrt(avg_nodes_2 - avg_nodes**2))
        stats["avg_nodes_per_batch"] = avg_nodes
        stats["std_nodes_per_batch"] = std_nodes

        return stats
    
    def print_stats(self):
        if self._total_passes == 0:
            print("No stats to report yet.")
            return
        avg_nodes = self._num_nodes / self._total_passes
        avg_nodes_2 = self._num_nodes_2 / self._total_passes
        std_nodes = (np.sqrt(avg_nodes_2 - avg_nodes**2))
        print(f"BatchClipper Stats over {self._total_passes} batches:")
        print(f"  Avg nodes per batch: {avg_nodes:.2f} +/- {std_nodes:.2f}")
        print(f"  Max nodes in a batch: {self._max_num_nodes}")
        print(f"  Min nodes in a batch: {self._min_num_nodes}")
        print(f"  Total clipped samples: {self._total_clipped}")
        print(f"  Total added from cache: {self._total_added}")
        print(f"  Total cache trim events: {self._trimmed_cache_events}")
        print(f"  Total samples trimmed from cache: {self._trimmed_cache_samples}")

    def reset(self):
        self.clear_stats()
        self.clear_cache()
    
    def clear_stats(self):
        self._total_clipped = 0
        self._total_added = 0
        self._total_passes = 0
        self._max_num_nodes = 0
        self._min_num_nodes = float('inf')
        self._num_nodes = 0
        self._num_nodes_2 = 0 # squared
        self._trimmed_cache_events = 0
        self._trimmed_cache_samples = 0

    def clear_cache(self):
        #call this at the end of every epoch
        self.cache = []
        self.cache_sizes = []

    def cache_len(self):
        return len(self.cache)
    
    def cache_nodes(self):
        return sum(self.cache_sizes)
    
    def add_to_cache(self, data_list):
        # add clipped data to the cache

        data_sizes = [self.sample_size_fn(data) for data in data_list]
        self.cache.extend(data_list)
        self.cache_sizes.extend(data_sizes)
        # limit cache size - remove oldest samples if over max_cache_samples
        if len(self.cache) > self.max_cache_samples:
            self._trimmed_cache_events += 1
            self._trimmed_cache_samples += len(self.cache) - self.max_cache_samples
            
            if self.verbose:
                print(f"Cache size exceeded max_cache_samples ({self.max_cache_samples}). Trimming oldest samples.")
            self.cache = self.cache[-self.max_cache_samples:]
            self.cache_sizes = self.cache_sizes[-self.max_cache_samples:]
            
    
    def pull_from_cache(self, num_nodes_needed):
        # pull samples from cache until num_nodes_needed is met
        pulled_data_list = []
        pulled_node_count = 0
        if self.pull_method == 'random':
            sorted_indices = np.random.permutation(len(self.cache))
        elif self.pull_method == 'smallest':
            # sort cache indices by size ascending - add smallest samples first
            sorted_indices = sorted(np.arange(len(self.cache)), key=lambda i: self.cache_sizes[i])
        elif self.pull_method == 'largest':
            # sort cache indices by size descending - add largest samples first
            sorted_indices = sorted(np.arange(len(self.cache)), key=lambda i: self.cache_sizes[i], reverse=True)
        
        pulled_indices = []
        for idx in sorted_indices:
            if pulled_node_count + self.cache_sizes[idx] <= num_nodes_needed:
                # pulled_data_list.append(self.cache.pop(idx))
                # pulled_node_count += self.cache_sizes.pop(idx)
                pulled_data_list.append(self.cache[idx])
                pulled_node_count += self.cache_sizes[idx]
                pulled_indices.append(idx)
            else:
                continue
        
        # Remove items in reverse order to avoid index shifting
        for idx in sorted(pulled_indices, reverse=True):
            self.cache.pop(idx)
            self.cache_sizes.pop(idx)

        ####debug
        if self.verbose:
            print(f"Pulled {len(pulled_data_list)} samples from cache with total nodes: {pulled_node_count}")
        
        return pulled_data_list

    def batch_to_list(self, batch):
        ''' take in the batch output from a DataLoader and decompose it into a list of data samples '''
        
        return batch.to_data_list()

    def list_to_batch(self, data_list):
        ''' take in a list of data samples and reassemble into a batch '''
        return GraphBatch.from_data_list(data_list)

    def __call__(self, batch, use_cache=True):
        b_num_nodes = self.batch_size_fn(batch)

        if not self.active:
            # tracking only, no clipping or caching
            self.track(num_nodes=b_num_nodes, num_clipped=0, num_added=0)
            return batch

        if b_num_nodes > self.max_nodes:

            data_list = self.batch_to_list(batch)
            node_counts = [self.sample_size_fn(data) for data in data_list]

            if self.clip_method == 'random':
                # Sample data_list in random order and clip remaining samples to cache
                sorted_indices = np.random.permutation(len(data_list))
            elif self.clip_method == 'largest':
                # sort data_list indices by size ascending order - clip largest samples first
                sorted_indices = sorted(np.arange(len(data_list)), key=lambda i: node_counts[i])
            elif self.clip_method == 'smallest':
                # sort data_list indices by size descending order - clip smallest samples first
                sorted_indices = sorted(np.arange(len(data_list)), key=lambda i: node_counts[i], reverse=True)

            #split data_list into clipped and removed samples
            clipped_data_list = []
            removed_data_list = []
            total_nodes = 0
            for idx in sorted_indices:
                if total_nodes + node_counts[idx] <= self.max_nodes:
                    clipped_data_list.append(data_list[idx])
                    total_nodes += node_counts[idx]
                else:
                    removed_data_list.append(data_list[idx])

            #add removed samples to cache
            if use_cache:
                self.add_to_cache(removed_data_list)
            
            ### debug
            # print(f"Clipped batch to {len(clipped_data_list)} samples with total nodes: {total_nodes}")
            
            #track stats
            self.track(num_nodes=b_num_nodes, num_clipped=len(removed_data_list), num_added=0)
            
            #rebuild the batch
            batch = self.list_to_batch(clipped_data_list)
            return batch

        elif b_num_nodes < self.max_nodes:
            # print(f"Batch under max nodes ({self.max_nodes}). Attempting to pull from cache...")
            if use_cache and len(self.cache) > 0:
                # pull samples from the cache if available
                num_nodes_needed = self.max_nodes - b_num_nodes
                cached_data_list = self.pull_from_cache(num_nodes_needed)

                #add the cached samples to the batch
                data_list = self.batch_to_list(batch)
                data_list +=  cached_data_list
                # data_list = batch.to_data_list() +  cached_data_list

                #track stats
                self.track(num_nodes=b_num_nodes, num_clipped=0, num_added=len(cached_data_list))

                #rebuild the batch
                # batch = GraphBatch.from_data_list(data_list)
                batch = self.list_to_batch(data_list)
                return batch
            else: 
                #track stats
                self.track(num_nodes=b_num_nodes, num_clipped=0, num_added=0)
                return batch
        else:
            # print(f"Batch at max nodes ({self.max_nodes}). No clipping or pulling needed.")
            # batch is already at max nodes
            self.track(num_nodes=b_num_nodes, num_clipped=0, num_added=0)
            return batch            
        
class TupleBatchClipper(BatchClipper):
    '''
    A BatchClipper that works on batches of tuples (data, data, ...)

    useful for datasets that return multiple data objects per sample
    either as a tuple or list.
    eg: return data1, data2 from __getitem__

    Args:
        sample_size_index (int = 0): index of which of item in the tuple to use for node counting.
    '''
    def __init__(self, sample_size_index=0,**kwargs):
        super().__init__(**kwargs)

        self.sample_size_index = sample_size_index

        if kwargs.get('sample_size_fn') is None:
            self.sample_size_fn = lambda data: data[self.sample_size_index].num_nodes # default

        if kwargs.get('batch_size_fn') is None:
            self.batch_size_fn = lambda batch: batch[self.sample_size_index].num_nodes # default
    
    def batch_to_list(self, list_batch):
        ''' 
        take in the batch output from a DataLoader and decompose it into a list of data samples 
        *input must be a list of batched data objects*
        '''
        if isinstance(list_batch, GraphBatch):
            raise ValueError(f"Input batch must be a list of graph Batch objects, got {type(list_batch)}")

        if not isinstance(list_batch, list):
            raise ValueError(f"Input batch must be a list of graph Batch objects, got {type(list_batch)}")
            

        #list_batch = [batch1, batch2, ... ]
        #convert input to a list of data lists
        list_data_list = [b.to_data_list() for b in list_batch] # [ data_list1, data_list2, ... ]
        #convert to list of tuples data samples
        data_list = list(zip(*list_data_list) ) # [ (data1, data2, ...), (data1, data2, ...), ... ]
        return data_list

    def list_to_batch(self, data_list):
        ''' take in a list of data samples and reassemble into a batch '''
        
        list_of_lists = list(zip(*data_list))
        list_of_batches = [GraphBatch.from_data_list(datalist) for datalist in list_of_lists]
        return list_of_batches
        