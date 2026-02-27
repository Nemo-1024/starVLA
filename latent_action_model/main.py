from lightning.pytorch.cli import LightningCLI

from latent_action_model.core.lam_lightinng import VJEPA_LAM
from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule


cli = LightningCLI(
    VJEPA_LAM,
    LeRobotDataModule,
    seed_everything_default=2026,
    save_config_kwargs={"overwrite": True},
)
