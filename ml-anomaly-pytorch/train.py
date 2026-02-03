"""
Training script for the anomaly detection autoencoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import logging
from model.autoencoder import TransactionAutoencoder
from typing import Optional
from model.preprocessing import FeaturePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000) -> np.ndarray:
    """
    Generate synthetic transaction data for training.
    In production, this would load from a database.
    """
    np.random.seed(42)
    preprocessor = FeaturePreprocessor()

    features_list = []

    for _ in range(n_samples):
        # Generate random transaction features
        features = {
            "amount": np.random.exponential(500) + 10,
            "currency": np.random.choice(["BRL", "USD", "EUR", "GBP"]),
            "country": np.random.choice(["BR", "US", "UK", "DE", "FR"]),
            "device": np.random.choice(["mobile", "desktop", "tablet"]),
            "hour": np.random.randint(0, 24),
            "day_of_week": np.random.randint(0, 7),
        }

        processed = preprocessor.preprocess(features)
        features_list.append(processed)

    return np.array(features_list, dtype=np.float32)


def train_model(
    model: TransactionAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
) -> TransactionAutoencoder:
    """Train the autoencoder model"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Calculate threshold based on validation data
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            errors = model.reconstruction_error(batch_data)
            reconstruction_errors.extend(errors.cpu().numpy())

    # Set threshold as 95th percentile
    model.threshold = np.percentile(reconstruction_errors, 95)
    logger.info(f"Anomaly threshold set to: {model.threshold:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--output", type=str, default="model/trained_model.pt", help="Output model path"
    )
    args = parser.parse_args()

    logger.info("Generating synthetic training data...")
    data = generate_synthetic_data(n_samples=10000)

    # Split into train/val
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(train_data))
    val_dataset = TensorDataset(torch.tensor(val_data))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # Create and train model
    model = TransactionAutoencoder(input_dim=10, encoding_dim=4)

    logger.info("Starting training...")
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(trained_model.state_dict(), args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
