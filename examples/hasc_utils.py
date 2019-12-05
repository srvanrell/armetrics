# functions to be used with hasc dataset

import pandas as pd
import numpy as np
import scipy as sp

import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate, hamming
from scipy.fftpack import fft, fftshift, ifft

temp_values = set()
replacements = {"aspalt": "asphalt",
                "aspjalt": "asphalt",
                "bottomw": "bottom",
                "buttom": "bottom",
                "down stair": "down,stairs",
                "down,inpocket": "down,in pocket",
                "fixed": "fixed",
                "indoors": "indoor",
                "leftfront": "left-front",
                "outerouter": "outer",
                "pans": "pants",
                "position(fixed)": "fix",
                "rigft": "right",
                "rigth": "right",
                "rigtht": "right",
                "run, inpocket": "run,in pocket",
                "waer": "wear",
                "ware": "wear",
                "weist": "waist"
                }


def _parse_meta_line(line):
    line = line.strip()
    line = line.replace("：", ":")  # replace weird colons
    if ":" not in line:
        if line != '':
            print(repr(line))
        return None
    # Fixing common found errors
    line = line.replace("TerminalPosition:  TerminalPosition:", "TerminalPosition:")
    line = line.replace("sneakers:leather shoes", "sneakers;leather shoes")
    if line.count(":") > 1:
        line = line[:line.rfind("AttachmentDirection:")]

    name, values = line.split(":")

    # Fixing TerminalPosition and other misspelled values
    values = values.lower()
    for wrong, right in replacements.items():
        values = values.replace(wrong, right)
    values = [v.strip() for v in values.strip().split(";")]
    values.sort()
    temp_values.update(v + "\t" + name + "\n" for v in values)
    values = ";".join(values)  # values cannot be a list

    return [name, values]


def parse_meta(meta_path):
    with open(meta_path, 'r') as meta_stream:
        meta_dict = dict()
        for line in meta_stream.readlines():
            try:
                p_line = _parse_meta_line(line)
                if p_line:
                    meta_dict[p_line[0]] = p_line[1]
            except ValueError:
                print("Something failed reading ", meta_path, "\nFix this line >>>", line)
    return meta_dict

# TODO split read_acc functionalities into functions
def read_acc(acc_file_path, new_freq=100.0, initial_t_crop=0.0, ending_t_crop=0.0):
    """
    Load acc file, compute magnitude acc, and interpolates to new sample frequency new_freq.
    t_start and t_end given in seconds indicate how many seconds to discard.
    """
    acc = pd.read_csv(acc_file_path, names=["t", "x", "y", "z"], dtype=np.float64)
    acc.drop_duplicates("t", inplace=True)  # Avoid t duplicates. Actually, they shouldn't exist

    # Compute amag
    acc["mag"] = np.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

    # Interpolate linearly to given sample frequency
    new_acc = pd.DataFrame(np.arange(acc.t.min(), acc.t.max(), 1.0 / new_freq), columns=["t"])
    axes = ["x", "y", "z", "mag"]
    interpoler = interp1d(acc["t"], acc[axes], fill_value="extrapolate", axis=0)
    new_acc[axes] = pd.DataFrame(interpoler(new_acc["t"]))

    # Crop signal to given initial and ending times
    first_t = new_acc.t[0]
    last_t = new_acc.t[len(new_acc.t)-1]
    cropped_first_t = first_t + initial_t_crop
    cropped_last_t = last_t - ending_t_crop

    cropped_acc = new_acc[(cropped_first_t <= new_acc.t) & (new_acc.t <= cropped_last_t)]
    
    # print(1 / acc.t.diff().mean(), 1 / interpacc.t.diff().mean(), acc_file_path)  # Real frequency

    return cropped_acc.reset_index(drop=True)


def _label_parser(line):
    line = str(line)
    line = line.strip().lower()
    if ";" in line:
        line = line[:line.find(";")]

    misspelled_labels = {"down": "stdown",
                         "up": "stup",
                         "sudown": "stdown",
                         "stdwon": "stdown",
                         "stdowns": "stdown",
                         "run": "jog"}  # FIXME don't know if it is right
    for misspelled, correct in misspelled_labels.items():
        if line == misspelled:
            line = correct

    return line


def read_labels(label_file_path):
    """Load acc file, compute magnitude and fix time to start with 0"""
    labels = pd.read_csv(label_file_path, names=["t_ini", "t_end", "label"], comment="#")
    labels["label"] = labels["label"].apply(lambda l: _label_parser(l))

    labels = labels[labels["t_end"].notnull()]  # remove entries that correspond to instant events
    # Remove inadequate labels
    labels = labels[labels.label != "jump"]
    labels = labels[labels.label != "5633"]
    labels = labels[labels.label != "run"]
    labels = labels[labels.label != "situp"]

    return labels


def get_window_times(signal, window_size, window_overlap, start=0, end=None):
    """
    Split a signal into windows of given size and overlap.

    :param signal:
    :param window_size: given in samples
    :param window_overlap: given in samples
    :param start: default value is 0
    :param end: default value is len(signal)
    :return list of windowed signals
    """
    if not end:
        end = len(signal)
    frame_indexes = []
    while start + window_size <= end:
        frame_indexes.append((start, start+window_size, start+window_size/2))
        start = start + (window_size - window_overlap)

    return frame_indexes


def get_windows(signal, window_size, window_overlap, start=0, end=None):
    """
    Split a signal into windows of given size and overlap.

    :param signal:
    :param window_size: given in samples
    :param window_overlap: given in samples
    :param start: default value is 0
    :param end: default value is len(signal)
    :return list of windowed signals
    """
    frame_indexes = get_window_times(signal, window_size, window_overlap, start=start, end=end)

    return (signal[f_start:f_end] for f_start, f_end, f_center in frame_indexes)


def extract_features(frame, n, name="fft", fs=None):

    f_len = len(frame)
    hamm = hamming(f_len)

    if name == "fft":
        absfft = np.abs(fft(frame * hamm))
        features = absfft[:n]

    if name == "cep":
        f_centered = frame - np.mean(frame)
        f_w = f_centered * hamm

        abs_fft = np.abs(fft(f_w))
        log_fft = np.log(abs_fft + np.spacing(1))  # adds a tiny number to avoid zeros
        cep = ifft(log_fft).real

        features = cep[:n]

    if name == "time":
        f_ene = np.sum(frame ** 2)
        f_max = np.max(frame)
        f_min = np.min(frame)
        f_amp = f_max - f_min
        f_std = np.std(frame)

        features = np.array([f_ene, f_amp, f_max, f_min, f_std])

    if name == "period":
        t = np.arange(0.0, f_len / fs, 1 / fs)  # time
        corr, lags = xcorr(frame, norm="unbiased")
        zcorr = corr[f_len:int(f_len * 1.5)]
        amplitud = max(zcorr) - min(zcorr)
        zcorrcentrado = zcorr - np.mean(zcorr)
        diffz = np.diff(np.sign(zcorrcentrado))
        cruceporcero = np.where(diffz < 0)[0][:6]

        umbral_autocorr = 0.04  # las señales no correlacionadas presentaban amplitudes menores a ese umbral
        # si no supera el umbral para decidir que hay autocorrelacion
        if amplitud < umbral_autocorr:
            maxycorr = 0  # esto devuelve fp = 0
        else:
            # busca la posición del primer pico
            if len(cruceporcero) >= 2:
                maxycorr = np.where(zcorr == max(zcorr[cruceporcero[0]:cruceporcero[1]]))[0]
            elif len(cruceporcero) == 1:
                maxycorr = np.where(zcorr == max(zcorr[cruceporcero[0]:]))[0]
            else:
                maxycorr = 0

        fp = t[maxycorr]  # traduzco lo hallado en segundos

        if fp > 1:  # si es mayor a 1 segundo lo anulo (no pueden admito movimientos taan lentos)
            fp = 0.0

        features = np.array(fp, ndmin=1)

    return features


def get_label(labels, frame_center):
    for label_ini, label_end, label in labels:
        if label_ini <= frame_center <= label_end:
            return label
    return None


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "%d"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=2)
        fmt = "%0.2f"

    thresh = 2. * cm.max() / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt % cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
def plot_confusion_matrix_pr(cm, classes, yclasses, title='', 
                             recall=None, precision=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, yclasses)

    fmt = "%d"

    thresh = 2. * cm.max() / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            plt.text(j, i, fmt % cm[i, j],
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    j=11
    for i in range(cm.shape[0]):
        if recall[i] > 0:
            plt.text(j, i, "%0.2f" % recall[i],
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def xcorr(x, y=None, maxlags=None, norm='biased'):
    """Cross-correlation using numpy.correlate

    Estimates the cross-correlation (and autocorrelation) sequence of a random
    process of length N. By default, there is no normalisation and the output
    sequence of the cross-correlation has a length 2*N+1.

    :param array x: first data array of length N
    :param array y: second data array of length N. If not specified, computes the
        autocorrelation.
    :param int maxlags: compute cross correlation between [-maxlags:maxlags]
        when maxlags is not specified, the range of lags is [-N+1:N-1].
    :param str option: normalisation in ['biased', 'unbiased', None, 'coeff']

    The true cross-correlation sequence is

    .. math:: r_{xy}[m] = E(x[n+m].y^*[n]) = E(x[n].y^*[n-m])

    However, in practice, only a finite segment of one realization of the
    infinite-length random process is available.

    The correlation is estimated using numpy.correlate(x,y,'full').
    Normalisation is handled by this function using the following cases:

        * 'biased': Biased estimate of the cross-correlation function
        * 'unbiased': Unbiased estimate of the cross-correlation function
        * 'coeff': Normalizes the sequence so the autocorrelations at zero
           lag is 1.0.

    :return:
        * a numpy.array containing the cross-correlation sequence (length 2*N-1)
        * lags vector

    .. note:: If x and y are not the same length, the shorter vector is
        zero-padded to the length of the longer vector.

    .. rubric:: Examples

    .. doctest::

        >>> from spectrum import *
        >>> x = [1,2,3,4,5]
        >>> c, l = xcorr(x,x, maxlags=0, norm='biased')
        >>> c
        array([ 11.])

    .. seealso:: :func:`CORRELATION`.
    """
    N = len(x)
    if y is None:
        y = x
    assert len(x) == len(y), 'x and y must have the same length. Add zeros if needed'

    if maxlags is None:
        maxlags = N - 1
        lags = np.arange(0, 2 * N - 1)
    else:
        assert maxlags < N
        lags = np.arange(N - maxlags - 1, N + maxlags)
    assert maxlags <= N, 'maxlags must be less than data length'

    res = np.correlate(x, y, mode='full')

    if norm == 'biased':
        Nf = float(N)
        res = res[lags] / float(N)  # do not use /= !!
    elif norm == 'unbiased':
        res = res[lags] / (float(N) - abs(np.arange(-N + 1, N)))[lags]
    # elif norm == 'coeff':
    #     Nf = float(N)
    #     rms = rms_flat(x) * rms_flat(y)
    #     res = res[lags] / rms / Nf
    else:
        res = res[lags]

    lags = np.arange(-maxlags, maxlags + 1)
    return res, lags
