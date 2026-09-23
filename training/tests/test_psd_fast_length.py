import pytest

from ifv_training.psd_fast_length import fast_length_row, install_fast_psd_length


def test_fast_length_is_exact_template_length_without_touching_media():
    class PoisonMedia(dict):
        def __getitem__(self, key):
            raise AssertionError('length stage must not read or decode media')

    ids = [1, 2, 3, 4]
    row = {'input_ids': ids, 'topk': 20, 'psd_media': PoisonMedia(path='image')}
    assert fast_length_row(row, max_length=4) is row
    assert row['lengths'] == [len(ids)]
    with pytest.raises(ValueError, match='max_length'):
        fast_length_row(row, max_length=3)
    with pytest.raises(ValueError, match='topk'):
        fast_length_row({'input_ids': ids, 'topk': 0}, max_length=4)


def test_patch_only_changes_psd_length_pass():
    class AddLengthPreprocessor:
        def preprocess(self, row):
            row['original_called'] = True
            return row

    class IfvPsdTopKTemplate:
        max_length = 8

    class OtherTemplate:
        pass

    install_fast_psd_length(AddLengthPreprocessor)
    install_fast_psd_length(AddLengthPreprocessor)
    preprocessor = AddLengthPreprocessor()
    preprocessor.template = IfvPsdTopKTemplate()
    row = preprocessor.preprocess({'input_ids': [1, 2], 'topk': 20})
    assert row == {'input_ids': [1, 2], 'topk': 20, 'lengths': [2]}
    preprocessor.template = OtherTemplate()
    assert preprocessor.preprocess({}) == {'original_called': True}
