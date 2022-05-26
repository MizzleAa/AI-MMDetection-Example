data_root = "./data/balloon"

data = dict(
    train=dict(
        ann_file=f"{data_root}/train/json/train.json",
        img_prefix=f"{data_root}/train/image/",
    ),
    val=dict(
        ann_file=f"{data_root}/val/json/val.json",
        img_prefix=f"{data_root}/val/image/",
    ),
    test=dict(
        ann_file=f"{data_root}/test/json/test.json",
        img_prefix=f"{data_root}/test/image/",
    )
)