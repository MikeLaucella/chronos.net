"""
trainer.py

The training script for Chronos models
"""

import logging

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb

from chronos.model.factory import get_model
from chronos.data.chrono_module import ChronosDataModule


def run_training(args: dict):
    """Run a training session for testing purposes."""

    logger = logging.getLogger('chronos.trainer')

    # Initialize wandb run
    logger.info('Initializing wandb run.')

    wandb.login()
    run = wandb.init(project='Chronos.Train', job_type='train', config=args)

    logger.info('Initializing data module and model.')

    # Initialize data module
    dm = ChronosDataModule(
        images={'naip_hist', 'eros_hist'},
        masks={'masks'},
        zarr_dir=args.zarr_dir,
        batch_size=args.batch_size,
        accumulate=args.accumulate,
        workers=args.workers)

    dm.setup()

    # Initialize model
    model = get_model({
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr
    }, dm.labels)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='Chronos.Train', job_type='train')

    # Initialize Callbacks
    checkpoint_metric = "{val/loss:.2f}"
    checkpoint_file = f"{run.name}-{checkpoint_metric}"

    lr_callback = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, # saves top-K checkpoints based on "val_loss" metric
        monitor="val/loss",
        mode="min",
        dirpath=args.model_dir,
        filename=checkpoint_file,
    )

    logger.info('Starting training session.')

    # Initialize a trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        precision='16-mixed',
        fast_dev_run=args.dev_run,
        accumulate_grad_batches=args.accumulate,
        callbacks=[checkpoint_callback, lr_callback])

    # Train the model
    if not args.dry_run:
        trainer.fit(model, dm)
        trainer.save_checkpoint(f"{args.model_dir}/{run.name}-final.ckpt")

    # Optionally test the model
    if args.run_tests:
        logger.info('Running test dataloader through post training.')
        trainer.test(dataloaders=dm.test_dataloader())

    # Close wandb run
    run.finish()


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Run AerMAE training session.')
    parser.add_argument('--meta_file', type=str, required=True,
                        help='Path to the directory containing city metadata.')
    parser.add_argument('--zarr_dir', type=str, required=True,
                        help='Path to the directory containing zarr data.')
    parser.add_argument('--model_dir', type=str, required=False, default='./checkpoints',
                        help='Path to the directory for storing checkpoint data.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=6e-5,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--model', type=str, default='mb16',
                        help='Model architecture to use (e.g., mb16).')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup epochs for the learning rate scheduler.')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for the scheduler.')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='Number of batches to accumulate gradients over.')
    parser.add_argument('--dev_run', action='store_true',
                        help='If set, runs a quick development run.')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, runs a dry run without actual execution.')
    parser.add_argument('--run_tests', action='store_true',
                        help='If set, runs the test dataloader through post training.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of workers to use for the dataloaders.')
    parser.add_argument('--description', type=str, default='NA',
                        help='A description of the training experiment')

    load_dotenv() # take environment variables from .env.

    run_training(parser.parse_args())
