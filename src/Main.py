import os
from milvus import Milvus, IndexType, MetricType, Status
from src.extract_feature import VGGNet
from diskcache import Cache


def createCollection(model):
    param = {'collection_name': 'test01', 'dimension': 128, 'index_file_size': 1024, 'metric_type': MetricType.L2}
    client.create_collection(param)
    ivf_param = {'nlist': 16384}
    client.create_index('test01', IndexType.IVF_FLAT, ivf_param)
    names = os.listdir('pictures')
    imgvecs = [[model.extract_feat(pic)] for pic in names]
    status, ids = client.insert(collection_name='test01', records=imgvecs)
    for i in range(len(names)):
        cache[ids[i]] = names[i]


def Search(model):
    imgfind = input("Please enter name the image:\n")
    findvec = model.extract_feat("pictures/" + imgfind)
    search_param = {'nprobe': 16}
    status, results = client.search(collection_name='test01', query_records=findvec, top_k=3, params=search_param)
    return query_name_from_ids(results)


def query_name_from_ids(vids):
    res = []
    for i in vids:
        if i in cache:
            res.append(cache[i])
    return res


def main():
    model = VGGNet()
    createCollection(model)
    Search(model)


if __name__ == '__main__':
    client = Milvus(host='localhost', port='19530')
    cache = Cache(directory='pictures')
    main()
    client.close()
