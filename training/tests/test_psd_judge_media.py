import base64
import hashlib
import io

from PIL import Image
import pytest

from ifv_training.psd_judge_media import judge_image_transport
from ifv_training.psd_gemini_judge import review_images


def encoded(fmt, mode='RGB', **kwargs):
    output = io.BytesIO()
    Image.new(mode, (9, 7), (10, 50, 90, 120) if mode == 'RGBA' else (10, 50, 90)).save(
        output, format=fmt, **kwargs)
    return output.getvalue()


@pytest.mark.parametrize('fmt,mime', [('PNG','image/png'),('JPEG','image/jpeg'),
    ('WEBP','image/webp'),('GIF','image/gif')])
def test_accepted_media_is_byte_identical(fmt, mime):
    blob = encoded(fmt)
    block, record = judge_image_transport(blob)
    assert record is None and block['mime_type'] == mime
    assert base64.b64decode(block['data']) == blob


@pytest.mark.parametrize('mode', ['RGB','RGBA'])
def test_avif_transport_preserves_decoded_pixels_and_original_identity(mode):
    blob = encoded('AVIF', mode)
    block, record = judge_image_transport(blob)
    wire = base64.b64decode(block['data'])
    assert block['mime_type'] == 'image/png'
    assert record['source_sha256'] == hashlib.sha256(blob).hexdigest()
    assert record['transport_sha256'] == hashlib.sha256(wire).hexdigest()
    with Image.open(io.BytesIO(blob)) as source, Image.open(io.BytesIO(wire)) as target:
        assert source.mode == target.mode == mode and source.size == target.size
        assert source.tobytes() == target.tobytes()
        assert record['decoded_pixels_sha256'] == hashlib.sha256(source.tobytes()).hexdigest()
    assert judge_image_transport(blob) == (block, record)


def test_avif_transport_preserves_exif_and_color_profile():
    exif = Image.Exif(); exif[274] = 6
    blob = encoded('AVIF', exif=exif, icc_profile=b'synthetic-profile-for-roundtrip')
    block, _ = judge_image_transport(blob)
    with Image.open(io.BytesIO(blob)) as source, Image.open(io.BytesIO(base64.b64decode(block['data']))) as target:
        assert dict(source.getexif()) == dict(target.getexif())
        assert source.info.get('icc_profile') == target.info.get('icc_profile')


def test_animation_is_not_silently_reduced_to_one_frame():
    output = io.BytesIO()
    Image.new('RGB',(9,7),'red').save(output,format='AVIF',save_all=True,
        append_images=[Image.new('RGB',(9,7),'blue')],duration=100)
    with pytest.raises(ValueError, match='AVIF layout'):
        judge_image_transport(output.getvalue())


def test_unsupported_and_corrupt_images_still_fail():
    with pytest.raises(ValueError, match='unsupported'):
        judge_image_transport(encoded('BMP'))
    with pytest.raises(Exception):
        judge_image_transport(b'not an image')


def test_review_retains_source_hash_and_deduplicates_original_avif(tmp_path, monkeypatch):
    blob = encoded('AVIF'); digest = hashlib.sha256(blob).hexdigest()
    path = tmp_path/'task.avif'; path.write_bytes(blob)
    block = {'type':'image_url','image_url':{'url':'data:image/avif;base64,'+base64.b64encode(blob).decode()}}
    trace = {'state':{'runtime_case':{'image_sha256':digest},'runtime_store':{'runtime_path':'archive'},
        'all_steps':[{'metadata':{'context_request_id':'1'}}]}}
    monkeypatch.setattr('src.orchestrator.runtime_events.reconstruct_archived_request',lambda *args: {'input_payload':[block,block]})
    images, media = review_images({'source':trace},image_path=path)
    assert path.read_bytes() == blob and media['task_image_sha256'] == digest
    assert media['policy_images'][0]['image_sha256'] == [digest,digest]
    assert len(media['transport_conversions']) == 1
    assert len([x for x in images if x['type']=='image']) == 1
    assert media['transport_conversions'][0]['transport_mime'] == 'image/png'
