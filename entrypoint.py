import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name=None)
def main(cfg: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    if cfg.exp_name == "weekly_ablation":
        from examples.weekly_ablation import main as weekly_ablation
        return weekly_ablation(cfg)
    elif cfg.exp_name == "test_weekly_ablation":
        from examples.weekly_ablation import test as test_weekly_ablation
        return test_weekly_ablation(cfg)
    elif cfg.exp_name == "procthor":
        from examples.train_procthor import main as train_procthor
        return train_procthor(cfg)


if __name__ == "__main__":
    main()
