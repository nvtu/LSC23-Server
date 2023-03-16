import connect
import torch
import argparse
from pymilvus import (
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
    utility
)
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_path', type=str, help='Path to feature file')
    parser.add_argument('index_path', type=str, help='Path to index file')
    parser.add_argument('--collection_name', type=str, default='LSC23_CLIP')
    parser.add_argument('--model_name', type=str, default='CLIP-ViT-L/14')

    args = parser.parse_args()
    return args


def get_dims(model_name: str) -> int:
    if model_name == "CLIP-ViT-L/14":
        return 768
    return 0


def create_collection(collection_name: str, model_name: str, recreate: bool = True):
    if recreate:
        if utility.has_collection(collection_name, using=connect.MILVUS_ALIAS):
            utility.drop_collection(collection_name, using=connect.MILVUS_ALIAS)

    # Create collection
    collection_schema = CollectionSchema(
        fields=[
            FieldSchema(name='_id', dtype=DataType.VARCHAR, max_length=255, is_primary=True),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=get_dims(model_name)),
        ],
        description=collection_name,
    )

    # Create collection
    collection = Collection(
        name=collection_name, 
        schema=collection_schema,
        using=connect.MILVUS_ALIAS,
        shards_num=10,
        consistency_level='Strong'
    )

    print(f'Collection {collection} created!!!')


def create_collection_index(collection_name: str):
    collection = Collection(name=collection_name, using=connect.MILVUS_ALIAS)
    collection.create_index(
        field_name='embedding',
        index_params={
            'index_type': 'IVF_FLAT',
            'metric_type': 'IP',
            'params': {
                'nlist': 4096
            }
        }
    )

    print(f'Collection {collection} index created!!!')


def insert_embedding_data(data_path: str, index_path: str, collection_name: str):
    data = torch.load(open(data_path, 'rb')).cpu().numpy()
    indices = torch.load(open(index_path, 'rb'))

    BATCH_SIZE = 1000
    collection = Collection(name=collection_name, using=connect.MILVUS_ALIAS)

    n_data = len(data)
    for i in tqdm(range(0, n_data, BATCH_SIZE)):
        j = min(i + BATCH_SIZE, n_data)
        _indices = indices[i:j]	
        # Remove prefix /mnt/4TBSSD/nvtu/LSC23_Data_Mount/LSC23_highres_images with empty string
        PREFIX = '/mnt/4TBSSD/nvtu/LSC23_Data_Mount/LSC23_highres_images/'
        EXT = '.jpg'
        _indices = list(map(lambda x: x.replace(PREFIX, '').replace(EXT, ''), _indices))
        # --------------------------------------------------------
        _data = data[i:j]

        msg = collection.insert([_indices, _data])
        print(msg)


if __name__ == '__main__':
    args = get_args()
    connect.connect_milvus()
    # create_collection(args.collection_name, args.model_name)
    # create_collection_index(args.collection_name)
    insert_embedding_data(args.feature_path, args.index_path, args.collection_name)