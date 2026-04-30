import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURES
# ─────────────────────────────────────────────

def build_model(input_shape: tuple, units: list = None, dropout_rate: float = 0.2) -> Model:
    """
    Stacked LSTM with BatchNormalization and configurable layer sizes.

    Args:
        input_shape  : (time_steps, n_features)
        units        : list of LSTM unit sizes per layer, e.g. [128, 64, 32]
        dropout_rate : dropout applied after every LSTM layer
    """
    if units is None:
        units = [128, 64, 32]

    inputs = Input(shape=input_shape)
    x = inputs

    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)           # all but last return sequences
        x = LSTM(u, return_sequences=return_seq,
                 kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    x = Dense(32, activation="relu")(x)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def build_bidirectional_model(input_shape: tuple, dropout_rate: float = 0.2) -> Model:
    """
    Bidirectional LSTM — captures both forward and backward temporal patterns.
    Useful when sales data has strong seasonal symmetry.
    """
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(32))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation="relu")(x)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────
#  TRAINING CALLBACKS
# ─────────────────────────────────────────────

def get_callbacks(checkpoint_path: str = "best_model.keras", patience: int = 5):
    """
    Standard training callbacks:
      - EarlyStopping   : stops when val_loss stops improving
      - ReduceLROnPlateau: halves LR after 3 stagnant epochs
      - ModelCheckpoint : saves the best weights automatically
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


# ─────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────

def predict_inverse(model, X: np.ndarray, scaler, n_features: int) -> np.ndarray:
    """
    Run model.predict and inverse-transform the scaled predictions back to
    original sales units.

    Args:
        model     : trained Keras model
        X         : sequence array of shape (n, time_steps, n_features)
        scaler    : fitted MinMaxScaler
        n_features: total number of feature columns (used to rebuild dummy matrix)

    Returns:
        1-D numpy array of predicted sales in original scale
    """
    preds = model.predict(X, verbose=0)
    dummy = np.zeros((len(preds), n_features))
    dummy[:, 0] = preds[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]


def inverse_transform_targets(y: np.ndarray, scaler, n_features: int) -> np.ndarray:
    """Inverse-transform ground-truth target values."""
    dummy = np.zeros((len(y), n_features))
    dummy[:, 0] = y
    return scaler.inverse_transform(dummy)[:, 0]
