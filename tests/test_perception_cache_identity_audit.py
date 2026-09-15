from scripts.server.audit_perception_cache_identity import summarize


def row(case, image, cached=True):
    return {'case_id': case, 'image_sha256': image, 'cache_hit': cached,
            'tool': 'perceive_scene', 'result_sha256': 'same-result'}


def test_same_image_reuse_not_reported_as_cross_image():
    assert summarize([row('a', 'image1'), row('b', 'image1')])['collision_groups'] == 0


def test_cross_image_reuse_reports_cached_cases_not_original_producer():
    result = summarize([row('a', 'image1', False), row('b', 'image2'), row('c', 'image3')])
    assert result['collision_groups'] == 1
    assert result['cache_hit_cases_in_cross_image_groups'] == 2


def test_equal_uncached_outputs_alone_are_not_cache_contamination_proof():
    assert summarize([row('a', 'image1', False), row('b', 'image2', False)])['collision_groups'] == 0
