import gc

import faiss
import numpy as np
from tqdm import tqdm


def precise_dist(feat, nbrs, num_process=4, sort=True, verbose=False):
    import torch

    feat_share = torch.from_numpy(feat).share_memory_()
    nbrs_share = torch.from_numpy(nbrs).share_memory_()
    dist_share = torch.zeros_like(nbrs_share).share_memory_()

    precise_dist_share_mem(
        feat_share,
        nbrs_share,
        dist_share,
        num_process=num_process,
        sort=sort,
        verbose=verbose,
    )

    del feat_share
    gc.collect()
    return dist_share.numpy(), nbrs_share.numpy()


def precise_dist_share_mem(
    feat, nbrs, dist, num_process=16, sort=True, process_unit=4000, verbose=False
):
    from torch import multiprocessing as mp

    num, _ = feat.shape
    num_per_proc = int(num / num_process) + 1

    processes = []
    for pi in range(num_process):
        sid = pi * num_per_proc
        eid = min(sid + num_per_proc, num)
        p = mp.Process(
            target=bmm,
            kwargs={
                "feat": feat,
                "nbrs": nbrs,
                "dist": dist,
                "sid": sid,
                "eid": eid,
                "sort": sort,
                "process_unit": process_unit,
                "verbose": verbose,
            },
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def bmm(feat, nbrs, dist, sid, eid, sort=True, process_unit=4000, verbose=False):
    import torch

    _, cols = dist.shape
    batch_sim = torch.zeros((eid - sid, cols), dtype=torch.float32)
    for s in tqdm(range(sid, eid, process_unit), desc="bmm", disable=not verbose):
        e = min(eid, s + process_unit)
        query = feat[s:e].unsqueeze(1)
        gallery = feat[nbrs[s:e]].permute(0, 2, 1)
        batch_sim[s - sid : e - sid] = torch.bmm(query, gallery).view(-1, cols)

    if sort:
        sort_unit = int(1e6)
        batch_nbr = nbrs[sid:eid]
        for s in range(0, batch_sim.shape[0], sort_unit):
            e = min(s + sort_unit, eid)
            batch_sim[s:e], indices = torch.sort(batch_sim[s:e], descending=True)
            batch_nbr[s:e] = torch.gather(batch_nbr[s:e], 1, indices)
        nbrs[sid:eid] = batch_nbr
    dist[sid:eid] = 1.0 - batch_sim


def faiss_search_knn(
    feat,
    k,
    metric,
    nprobe=128,
    num_process=4,
    is_precise=True,
    sort=True,
    verbose=False,
):
    if metric == "cosine":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
        # TODO: Make sure all vectors are normalised in this case
        raise NotImplementedError(
            "Cosine distance is not implemented in faiss_search_knn"
        )
    else:
        faiss_metric = faiss.METRIC_L2
    dists, nbrs = faiss_search_approx_knn(
        query=feat,
        target=feat,
        k=k,
        nprobe=nprobe,
        verbose=verbose,
        metric=faiss_metric,
    )

    if is_precise:
        print("compute precise dist among k={} nearest neighbors".format(k))
        dists, nbrs = precise_dist(
            feat, nbrs, num_process=num_process, sort=sort, verbose=verbose
        )

    return dists, nbrs


class faiss_index_wrapper:
    def __init__(
        self,
        target,
        nprobe=128,
        index_factory_str=None,
        verbose=False,
        mode="proxy",
        using_gpu=True,
        metric=faiss.METRIC_INNER_PRODUCT,
    ):
        import faiss

        self._res_list = []

        num_gpu = faiss.get_num_gpus()
        print("[faiss gpu] #GPU: {}".format(num_gpu))

        size, dim = target.shape
        assert size > 0, "size: {}".format(size)
        index_factory_str = (
            "IVF{},PQ{}".format(min(8192, 16 * round(np.sqrt(size))), 32)
            if index_factory_str is None
            else index_factory_str
        )
        cpu_index = faiss.index_factory(dim, index_factory_str, metric)
        cpu_index.nprobe = nprobe

        if mode == "proxy":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False

            index = faiss.IndexProxy()
            for i in range(num_gpu):
                res = faiss.StandardGpuResources()
                self._res_list.append(res)
                sub_index = (
                    faiss.index_cpu_to_gpu(res, i, cpu_index, co)
                    if using_gpu
                    else cpu_index
                )
                index.addIndex(sub_index)
        elif mode == "shard":
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
            co.shard = True
            index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=num_gpu)
        else:
            raise KeyError("Unknown index mode")

        index = faiss.IndexIDMap(index)
        index.verbose = verbose

        # get nlist to decide how many samples used for training
        nlist = int(
            [item for item in index_factory_str.split(",") if "IVF" in item][0].replace(
                "IVF", ""
            )
        )

        # training
        if not index.is_trained:
            indexes_sample_for_train = np.random.randint(0, size, nlist * 256)
            index.train(target[indexes_sample_for_train])

        # add with ids
        target_ids = np.arange(0, size)
        index.add_with_ids(target, target_ids)
        self.index = index

    def search(self, *args, **kargs):
        return self.index.search(*args, **kargs)

    def __del__(self):
        self.index.reset()
        del self.index
        for res in self._res_list:
            del res


def batch_search(index, query, k, bs, verbose=False):
    n = len(query)
    dists = np.zeros((n, k), dtype=np.float32)
    nbrs = np.zeros((n, k), dtype=np.int64)

    for sid in tqdm(range(0, n, bs), desc="faiss searching...", disable=not verbose):
        eid = min(n, sid + bs)
        dists[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], k)
    return dists, nbrs


def faiss_search_approx_knn(
    query,
    target,
    k,
    nprobe=128,
    bs=int(1e6),
    index_factory_str=None,
    verbose=False,
    metric=faiss.METRIC_INNER_PRODUCT,
):
    index = faiss_index_wrapper(
        target,
        nprobe=nprobe,
        index_factory_str=index_factory_str,
        verbose=verbose,
        metric=metric,
    )
    dists, nbrs = batch_search(index, query, k=k, bs=bs, verbose=verbose)

    del index
    gc.collect()
    return dists, nbrs
