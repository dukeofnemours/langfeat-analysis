import numpy as np

from langfeat_analysis.preprocessing.audio import (
    AudioPreprocessData,
    AudioPreprocessedData,
    build_tr_aligned_stimulus,
    z_score_features,
)


def test_z_score_features_standardizes_each_feature_across_time():
    data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    standardized = z_score_features(data)

    np.testing.assert_allclose(standardized.mean(axis=1), [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(standardized.std(axis=1), [1.0, 1.0])


def test_audio_record_stores_standardized_data_and_zeroes_constant_features():
    record = build_tr_aligned_stimulus(
        "stimulus",
        "mfcc",
        np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]]),
        ["mfcc.0", "mfcc.1"],
    )

    assert isinstance(record, AudioPreprocessData)
    assert AudioPreprocessData is AudioPreprocessedData
    np.testing.assert_allclose(record.st_data[0], [-1.224744871, 0.0, 1.224744871])
    assert record.st_data[1] == [0.0, 0.0, 0.0]
    assert record.data == [[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]]
