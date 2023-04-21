import os, json
from detectron2.data import DatasetCatalog, MetadataCatalog

def loadJson(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def prepare_sor_dataset_list_of_dict(dataset_name, split, root="datasets"):
    path_to_ds = os.path.join(root, dataset_name, "{}_{}.json".format(dataset_name, split))
    print("Path to {}: {}".format(dataset_name, path_to_ds), flush=True)

    dataset = loadJson(path_to_ds)
    print(dataset["__comment__"], flush=True)

    image_path = os.path.join(root, dataset_name, "images", split)
    dataset_data = dataset["data"]
    for i in range(len(dataset_data)):
        dataset_data[i]["file_name"] = os.path.join(image_path, dataset_data[i]["image_name"]+".jpg")
    print("#Length of SOR dataset [{}]:{}".format(dataset_name, len(dataset_data)), flush=True)

    return dataset_data

def register_sor_dataset(cfg):
    DatasetCatalog.register("assr_train", lambda s="train":prepare_sor_dataset_list_of_dict(dataset_name="assr", split=s, root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("assr_val", lambda s="val":prepare_sor_dataset_list_of_dict(dataset_name="assr", split=s, root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("assr_test", lambda s="test":prepare_sor_dataset_list_of_dict(dataset_name="assr", split=s, root=cfg.DATASETS.ROOT))

    # DatasetCatalog.register("irsr_train", lambda s="train":prepare_sor_dataset_list_of_dict(dataset_name="irsr", split=s, root=cfg.DATASETS.ROOT))
    # DatasetCatalog.register("irsr_test", lambda s="test":prepare_sor_dataset_list_of_dict(dataset_name="irsr", split=s, root=cfg.DATASETS.ROOT))
    