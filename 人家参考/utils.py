import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(filenames):
    """ load csv file to pd.DataFrame
    args:
        fielnames: list,
    returns:
        data: pd.DataFrame
    """
    data = [pd.read_csv(filename) for filename in tqdm(filenames)]
    return pd.concat(data)


def preprocessing(data, has_label=True):
    """ preprocessing data, do some transform
    """
    # band freq
    data['f'] = data['Frequency Band'].apply(
        lambda x: 33.9 * np.log10(x))

    # cell height
    data['h_b'] = data['Height'].apply(
        lambda x: -13.82 * np.log10(x + 1))

    # cell clutter
    data['Clu_c'] = data['Cell Clutter Index']

    # rec clutter
    data['Clu_t'] = data['Clutter Index']

    data['delta_d_h'] = ((data['Cell X'] - data['X']) ** 2 + \
                         (data['Cell Y'] - data['Y']) ** 2) ** 0.5

    # rec relative height
    theta = (data['Electrical Downtilt'] + data['Mechanical Downtilt']) / 90
    relative_height = data['Height'] + data['Cell Altitude'] - data['Altitude']
    flag = ((relative_height / np.tan(theta)) > data['delta_d_h']).astype(np.float32)
    data['delta_d_v'] = (relative_height - data['delta_d_h'] * np.tan(theta)) * flag

    # theta
    data['theta'] = theta

    # horizontal distance dot building height
    data['f_h_d'] = - 6.55 * np.log10(data['Building Height'] + 1) * \
                        np.log10(data['delta_d_h'] + 1)

    # power decay
    a_square = ((data['Cell X'] - data['X']) ** 2 + \
                (data['Cell Y'] + 1 - data['Y']) ** 2)
    # (a^2 + b^2 - c^2) / (2 * a * b)
    cos_beta = ((1**2 + data['delta_d_h']**2) - a_square) / \
                           (2 * data['delta_d_h'] * 1)
    beta = np.arccos(cos_beta)
    data['beta'] = beta # magic feature to improve pcrr
    beta += (data['Azimuth'] / 90)

    data['f_decay'] = np.abs(np.cos(beta) * data['Frequency Band'])

    # distance horizontal
    data['delta_d_h'] = data['delta_d_h'].apply(
        lambda x: 44.9 *np.log10(x + 1))

    # data['height_diff'] = data['Cell Altitude'] + data['Height'] - \
    #                       data['Altitude'] - data['Building Height']

    features = data[['f', 'h_b', 'Clu_c', 'Clu_t', 'delta_d_v',
                'f_h_d', 'f_decay', 'delta_d_h','beta',
                'Cell Building Height']]

    if has_label:
        return features, data['RSRP']
    else:
        return features


def pcrr(y_pred, dtrain):
    """ calculate PCRR
    args:
        y_pred: predict label
        dtrain: xgb.DMatrix
    returns:
        name: str, "pcrrc"
        pcrr: int, value of pcrr
    """
    labels = dtrain.get_label()
    y_pred = y_pred
    t = - 103
    tp = len(labels[(labels < t)  & (y_pred < t)])
    fp = len(labels[(labels >= t) & (y_pred < t)])
    fn = len(labels[(labels < t)  & (y_pred >= t)])
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    pcrr = 2 * (precision * recall) / (precision + recall + 1e-12)
    return "pcrr", pcrr
