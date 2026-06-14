from .dataset import PushTDataset


def make_pusht_dataset(
    name: str,
    zarr_path: str,
    shape_meta: dict,
    seq_len: int = 19,
    val_ratio: float = 0.02,
    max_train_episodes: int = 90,
    seed: int = 42,
    normalize_action: bool = True,
    normalize_lowdim: bool = True,
):
    train_set = PushTDataset(
        name=name,
        zarr_path=zarr_path,
        shape_meta=shape_meta,
        seq_len=seq_len,
        val_ratio=val_ratio,
        max_train_episodes=max_train_episodes,
        seed=seed,
        split="train",
        normalize_action=normalize_action,
        normalize_lowdim=normalize_lowdim,
    )
    val_set = train_set.get_validation_dataset()
    return train_set, val_set
