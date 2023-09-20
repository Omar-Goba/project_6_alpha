from sklearn.metrics import roc_auc_score, mean_squared_error as mse, log_loss as ll
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import xgboost as xgb   
import numpy as np
import os, warnings


def calc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
        calculates slew of scores and return them in a dictionary
            - auc
            - mse
            - logloss
        args:
            y_true: true labels
            y_pred: predicted labels
        returns:
            dictionary of scores
    """
    ### calculate the score ###
    scores = {
        "auc": roc_auc_score(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "ll": ll(y_true, y_pred),
    }

    return scores


def init_xgb() -> xgb.XGBClassifier:
    """
        builds the model with the best hyperparameters
        this is an independent function so that we can use it 
        for training and loading the model
        args:
            None
        returns:
            xgb model
    """
    ### init the model ###
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        missing=-999,
        random_state=42,
    )

    return model


def train_xgb(tr_x: np.ndarray, tr_y: np.ndarray, ts_x: np.ndarray, ts_y: np.ndarray, model: xgb.XGBClassifier) -> xgb.XGBClassifier:
    """
        trains the model and returns it
        args:
            tr_x: training data
            tr_y: training labels
            ts_x: testing data
            ts_y: testing labels
            model: xgb model
        returns:
            xgb model
    """
    ### train the model ###
    model.fit(
        tr_x, tr_y,
        eval_set=[(ts_x, ts_y)],
        eval_metric=["auc", "logloss"],
        verbose=100,
    )

    return model


def predict_xgb(ts_x: np.ndarray, model: xgb.XGBClassifier) -> np.ndarray:
    """
        predicts the labels
        this is an independent function so that we can use it
        in the ensemble
        args:
            ts_x: testing data
            model: xgb model
        returns:
            predictions
    """
    ### predict ###
    preds = model.predict_proba(ts_x)[:, 1]

    return preds


def save_xgb(model: xgb.XGBClassifier, scores: dict[str, float]) -> int:
    """
        saves the model
        will save the model with the following name:
            ts={timestamp}_loss={loss}
            where:
                timestamp: timestamp of the time when the model was saved (YYYYMMDDTHHMMSS)
                loss: sum of all the losses (l1 loss)
        args:
            model: xgb model
            scores: dictionary of scores
        returns:
            0 if successful
    """
    ### save the model ###
    root_path = "./src/models"
    time_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    l1_loss = sum(scores.values())
    name = f"xgb_ts={time_stamp}_loss={l1_loss:.4f}".replace(".", ",")

    model.save_model(f"{root_path}/{name}.model")

    return 0


def load_xgb(name: str = None) -> xgb.XGBClassifier:
    """
        loads the model with the given name
        if no name is given, it will load the best model
        using the l1 loss as the metric
        args:
            name: name of the model
        returns:
            xgb model
    """
    ### load the model ###
    root_path = "./src/models"
    model = init_xgb()
    if name is not None:
        model.load_model(f"{root_path}/{name}")
    else:
        all_models = [i for i in os.listdir(root_path) if ".gitkeep" not in i and "xgb" in i]
        bst = min(all_models, key=lambda x: float(x.split("_")[-1].split("=")[1].replace(",", ".").replace(".model", "")))
        model.load_model(f"{root_path}/{bst}")

    return model


def init_nn(tr_x: np.ndarray) -> keras.Model:
    """
        builds the model
        the model is not deep because the data is not that big
        this is done to prevent overfitting
        args:
            tr_x: training data
                to get the shape of the input
        returns:
            keras model
    """
    ### init the model ###
    inp = keras.layers.Input(shape=(tr_x.shape[1],))
    model = keras.layers.Dense(128, activation="relu")(inp)
    model = keras.layers.Dropout(0.2)(model)
    model = keras.layers.Dense(16, activation="relu")(model)
    model = keras.layers.Dropout(0.2)(model)
    out = keras.layers.Dense(1, activation="sigmoid")(model)

    model = keras.Model(inputs=inp, outputs=out)

    ### compile the model ###
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_nn(tr_x: np.ndarray, tr_y: np.ndarray, ts_x: np.ndarray, ts_y: np.ndarray, model: keras.Model) -> keras.Model:
    """
        trains the model and returns it
        args:
            tr_x: training data
            tr_y: training labels
            ts_x: testing data
            ts_y: testing labels
            model: keras model
        returns:
            keras model
    """
    ### train the model ###
    model.fit(
        tr_x, tr_y,
        validation_data=(ts_x, ts_y),
        epochs=10,
        batch_size=32,
        verbose=1,
    )

    return model


def predict_nn(ts_x: np.ndarray, model: keras.Model) -> np.ndarray:
    """
        predicts the labels
        this is an independent function so that we can use it
        in the ensemble
        args:
            ts_x: testing data
            model: keras model
        returns:
            predictions
    """
    ### predict ###
    preds = model.predict(ts_x)

    return preds


def save_nn(model: keras.Model, scores: dict[str, float]) -> int:
    """
        saves the model
        will save the model with the following name:
            ts={timestamp}_loss={loss}
            where:
                timestamp: timestamp of the time when the model was saved (YYYYMMDDTHHMMSS)
                loss: sum of all the losses (l1 loss)
        args:
            model: keras model
            scores: dictionary of scores
        returns:
            0 if successful
    """
    ### save the model ###
    root_path = "./src/models"
    time_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    l1_loss = sum(scores.values())
    name = f"nn_ts={time_stamp}_loss={l1_loss:.4f}".replace(".", ",")

    model.save(f"{root_path}/{name}.h5")

    return 0


def load_nn(name: str = None) -> keras.Model:
    """
        loads the model with the given name
        if no name is given, it will load the best model
        using the l1 loss as the metric
        args:
            name: name of the model
        returns:
            keras model
    """
    ### load the model ###
    root_path = "./src/models"
    if name is not None:
        model = keras.models.load_model(f"{root_path}/{name}")
    else:
        all_models = [i for i in os.listdir(root_path) if ".gitkeep" not in i and "nn" in i]
        bst = min(all_models, key=lambda x: float(x.split("_")[-1].split("=")[1].replace(",", ".").replace(".h5", "")))
        tf.keras.utils.disable_interactive_logging()
        model = keras.models.load_model(f"{root_path}/{bst}")

    return model


def main(show_pbar: bool = True, do_train: bool = True) -> int:
    """
        loads the data, trains/loads the following models
            - XGBoost
            - Neural Network
        and calculates the score and uses the weighted average
        to get the final predictions using the ensemble
        args:
            show_pbar: whether to show the progress bar or not
            do_train: whether to train the models or not
        returns:
            0 if successful
    """
    ### QoL ###
    warnings.filterwarnings("ignore")

    ### init the progress bar ###
    if show_pbar: pb = tqdm(total=16)

    ### load data ###
    if show_pbar: pb.set_description("Loading data")
    tr = np.load("./dbs/cooked/tr.npz")
    ts = np.load("./dbs/cooked/ts.npz")
    tr_x, tr_y = tr["x"], tr["y"]
    ts_x, ts_y = ts["x"], ts["y"]
    if show_pbar: pb.update()

    ### ~~~ XGBoost ~~~ ###
    ### init the model ###
    if show_pbar: pb.set_description("Init XGBoost")
    xgb_model = init_xgb()
    if show_pbar: pb.update()

    ### train the model ###
    if show_pbar: pb.set_description("Train/Load XGBoost")
    if do_train:
        xgb_model = train_xgb(tr_x, tr_y, ts_x, ts_y, xgb_model)
    else:
        xgb_model = load_xgb()
    if show_pbar: pb.update()

    ### predict ###
    if show_pbar: pb.set_description("Predict XGBoost")
    xgb_preds = predict_xgb(ts_x, xgb_model)
    if show_pbar: pb.update()

    ### calculate the score ###
    if show_pbar: pb.set_description("Calculate XGBoost score")
    xgb_scores = calc_score(ts_y, xgb_preds)
    if show_pbar: pb.update()

    ### save the model ###
    if show_pbar: pb.set_description("Save XGBoost")
    if do_train:
        save_xgb(xgb_model, xgb_scores)
    if show_pbar: pb.update()
    ### ~~~ XGBoost ~~~ ###

    ### ~~~ NN ~~~ ###
    ### init the model ###
    if show_pbar: pb.set_description("Init NN")
    nn_model = init_nn(tr_x)
    if show_pbar: pb.update()

    ### train the model ###
    if show_pbar: pb.set_description("Train/Load NN")
    if do_train:
        nn_model = train_nn(tr_x, tr_y, ts_x, ts_y, nn_model)
    else:
        nn_model = load_nn()
    if show_pbar: pb.update()

    ### predict ###
    if show_pbar: pb.set_description("Predict NN")
    nn_preds = predict_nn(ts_x, nn_model)
    if show_pbar: pb.update()

    ### calculate the score ###
    if show_pbar: pb.set_description("Calculate NN score")
    nn_scores = calc_score(ts_y, nn_preds)
    if show_pbar: pb.update()

    ### save the model ###
    if show_pbar: pb.set_description("Save NN")
    if do_train:
        save_nn(nn_model, nn_scores)
    if show_pbar: pb.update()
    ### ~~~ NN ~~~ ###

    ### ~~~ Ensemble ~~~ ###
    ### calculate the weights using the l2 norm ###
    if show_pbar: pb.set_description("Calculate weights")
    l2 = lambda x: sum(x**2)**0.5
    xgb_weight = l2(np.array(list(xgb_scores.values())))
    nn_weight = l2(np.array(list(nn_scores.values())))
    if show_pbar: pb.update()

    ### calculate the weights ###
    if show_pbar: pb.set_description("Calculate weights")
    xgb_weight = xgb_weight / (xgb_weight + nn_weight)
    nn_weight = nn_weight / (xgb_weight + nn_weight)
    if show_pbar: pb.update()

    ### calculate the weighted average ###
    if show_pbar: pb.set_description("Calculate weighted average")
    preds = xgb_weight * xgb_preds + nn_weight * nn_preds.reshape(-1)
    if show_pbar: pb.update()

    ### calculate the score ###
    if show_pbar: pb.set_description("Calculate ensemble score")
    scores = calc_score(ts_y, preds)
    tqdm.write(f"Ensemble score: {scores}")
    if show_pbar: pb.update()
    ### ~~~ Ensemble ~~~ ###

    ### save the predictions ###
    if show_pbar: pb.set_description("Save predictions")
    np.save("./dbs/cooked/preds.npy", preds)
    if show_pbar: pb.update()

    ### close the progress bar ###
    if show_pbar: pb.set_description("Done")
    if show_pbar: pb.close()

    return 0


if (__name__ == "__main__"):    main(do_train=True)
#   __   _,_ /_ __, 
# _(_/__(_/_/_)(_/(_ 
#  _/_              
# (/                
