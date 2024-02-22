from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything


def cli_main():
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    seed_everything(7, workers=True)
    cli_main()
