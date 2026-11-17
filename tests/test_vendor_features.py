import hashlib

import numpy as np
from moth_features import DEFAULT_FILTERS, load_mel_filters, log_mel

MEL_FILTERS_SHA256 = "4f2701b1d287d74a0dc9871026e9519d98cb76426615f2539b0d151a0ae4ec2e"


def test_vendored_filterbank_is_the_moth_release_asset():
    assert hashlib.sha256(DEFAULT_FILTERS.read_bytes()).hexdigest() == MEL_FILTERS_SHA256
    filters = load_mel_filters()
    assert filters.shape == (80, 201)
    assert filters.dtype == np.float32


def test_log_mel_uses_the_default_filterbank():
    t = np.arange(16000) / 16000
    mel = log_mel(0.1 * np.sin(2 * np.pi * 440 * t))
    assert mel.shape == (80, 100)
    assert np.isfinite(mel).all()
