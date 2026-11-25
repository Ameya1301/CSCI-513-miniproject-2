"""Train the in-lane classifier.

Input : bounding box [[xmin, ymin, xmax, ymax]] in pixels of the 600x600 camera image.
Output: probability that the detected car is in the ego lane (> 0.5 => in lane).

Run from inside mp2_distance_predictor/ (writes in_lane_model.h5 here).

Design notes
- Input scaling is a Keras `Normalization` layer adapted on the training set, so it is
  stored inside the .h5 and needs no custom code at load time (no Lambda layers).
- The data is ~93% in-lane. Out-of-lane frames get a larger class weight so the model
  actually learns them, but the weight is deliberately NOT fully balanced: calling an
  in-lane car "out of lane" hands the controller `inf` (dangerous), while the opposite
  mistake only makes it brake unnecessarily. So we keep a bias toward "in lane".
- Stratified train/validation split so the rare class appears in both.
"""

import numpy as np
import pandas as pd

import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

SEED = 513
MODEL_NAME = "in_lane_model"


def load_data(path='../dataset/annotations.csv'):
    df = pd.read_csv(path)
    df = df.dropna()  # Delete frames with no detections
    X = np.array([eval(bbox_str)[0] for bbox_str in df['bbox']], dtype=np.float32)
    y = np.isfinite(df['distance']).values.astype(np.float32)  # 1 = in lane
    return X, y


def build_model(X_train):
    norm = layers.Normalization(axis=-1, name="bbox_normalization")
    norm.adapt(X_train)

    model = keras.Sequential([
        keras.Input(shape=(4,)),
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ], name=MODEL_NAME)

    opt = keras.optimizers.Adam(learning_rate=0.001)
    # Loss spelled out in full: the shorthand fails to deserialize in the simulator.
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def report(model, X, y, name):
    p = model.predict(X, verbose=0)[:, 0]
    pred = p > 0.5
    truth = y > 0.5
    acc = (pred == truth).mean()
    # "missed lead": car truly in lane but predicted out of lane -> controller sees inf
    missed = (~pred & truth).sum()
    false_alarm = (pred & ~truth).sum()
    print(f"[{name}] acc={acc:.4f}  in-lane->out (dangerous)={missed}/{truth.sum()}  "
          f"out->in-lane={false_alarm}/{(~truth).sum()}")


def main():
    keras.utils.set_random_seed(SEED)
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    # Partial rebalancing (sqrt of the ratio), see design notes above.
    class_weight = {0: float(np.sqrt(n_pos / n_neg)), 1: 1.0}
    print(f"train: {int(n_pos)} in-lane, {int(n_neg)} out-of-lane; class_weight={class_weight}")

    model = build_model(X_train)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=60, mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5,
                          min_delta=1e-4, mode='min', verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000, batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks, verbose=2)

    report(model, X_train, y_train, "train")
    report(model, X_val, y_val, "val")

    model.save(f'{MODEL_NAME}.h5')
    print(f"saved {MODEL_NAME}.h5")

    # Sanity check: reload exactly like the simulator does and compare outputs.
    reloaded = keras.models.load_model(f'{MODEL_NAME}.h5')
    sample = X_val[:1]
    a = float(model(sample)[0, 0])
    b = float(reloaded(np.array(sample))[0, 0].numpy())
    assert abs(a - b) < 1e-5, (a, b)
    print(f"reload check OK ({a:.4f} == {b:.4f})")


if __name__ == '__main__':
    main()
