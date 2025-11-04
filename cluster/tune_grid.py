from ray import tune
from TrackSelectorDNN.configs.schema import load_config
from TrackSelectorDNN.tune.trainable import trainable

base_cfg = load_config("base.yaml").model_dump()

search_space = {
    **base_cfg,  # include all validated defaults
    "lr": tune.loguniform(1e-5, 1e-2),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "latent_dim": tune.choice([16, 32, 64]),
    "pooling_type": tune.choice(["sum", "mean", "softmax"]),
    "netA_hidden_layers": tune.choice([1, 2, 3, 4, 5]),
    "netB_hidden_layers": tune.choice([1, 2, 3, 4, 5]),
}

tuner = tune.Tuner(
    trainable,
    param_space=search_space,
    tune_config=tune.TuneConfig(metric="val_loss", mode="min", num_samples=1),
)

results = tuner.fit()

best_result = results.get_best_result(metric="val_loss", mode="min")

print("Best config found:")
print(best_result.config)

best_ckpt_dir = best_result.checkpoint
print("Best checkpoint path:", best_ckpt_dir)