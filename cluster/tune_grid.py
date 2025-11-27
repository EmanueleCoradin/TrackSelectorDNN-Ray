import os
os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "92"  # allow all trials to be pending

from ray import tune
from ray.air import RunConfig, CheckpointConfig

from TrackSelectorDNN.configs.schema import load_config
from TrackSelectorDNN.tune.trainable import trainable

base_cfg = load_config("base.yaml").model_dump()

search_space = {
    **base_cfg,  # include all validated defaults
    "epochs": 75,
    "patience": 15,
    "netB_activation": tune.choice(["relu", "leakyrelu", "silu",]),
    "netA_activation": tune.sample_from(lambda spec: spec.config["netB_activation"]),
    "lr": tune.loguniform(1e-5, 1e-2),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "latent_dim": tune.sample_from(
        lambda spec: 32 if spec.config["netB_hidden_dim"] >= 64 else 16
    ),


    "netA_hidden_dim": tune.choice([32, 64]),
    "netB_hidden_dim": tune.choice([32, 64, 128]),
    "pooling_type": tune.choice(["sum", "mean"]),
    "netA_hidden_layers": tune.choice([1, 2, 3]),
    "netB_hidden_layers": tune.choice([2, 3]),
    "lambda_sym": tune.loguniform(1e-4, 1e-1)

}

ray_storage_path = "/shared/ray_results"

run_config = RunConfig(
    name="track_selector_dnn_tuning",
    storage_path=ray_storage_path,  # logs and experiment metadata
    checkpoint_config=CheckpointConfig(
        checkpoint_score_attribute="val_loss",
        num_to_keep=50
    )
)

tuner = tune.Tuner(
    tune.with_resources(
        trainable,
        {"cpu": 1, "gpu": 1./50.}  # adjust CPU/GPU numbers as needed
    ),
    param_space=search_space,
    tune_config=tune.TuneConfig(metric="val_loss", mode="min", num_samples=1000),
    run_config=run_config,
)

results = tuner.fit()

best_result = results.get_best_result(metric="val_loss", mode="min")

print("Best config found:")
print(best_result.config)

best_ckpt_dir = best_result.checkpoint
print("Best checkpoint path:", best_ckpt_dir)