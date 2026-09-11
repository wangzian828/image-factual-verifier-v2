"""Read-only comparison of legacy perception images and complete-episode images."""
import argparse
from collections import Counter
from pathlib import Path

from prepare_full_data import rows, write, dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.package.resolve(strict=True)
    data = {}
    for kind in ('policy','perception'):
        index = rows(source/f'ms-swift-{kind}/index.jsonl')
        entries = {}
        for split in ('train','validation','test'):
            metadata = [r for r in index if r['split']==split]
            content = rows(source/f'ms-swift-{kind}/{split}.jsonl')
            assert len(metadata)==len(content)
            for meta, row in zip(metadata,content):
                entries[meta['episode_id']] = [Path(p).name for p in row['images']]
        data[kind] = entries
    policy = {p for names in data['policy'].values() for p in names}
    perception = {p for names in data['perception'].values() for p in names}
    delivery = {p.name for p in (source/'images').iterdir()}
    shared_episodes = set(data['policy']) & set(data['perception'])
    legacy_only_episodes = set(data['perception']) - set(data['policy'])
    legacy_only_images = {p for e in legacy_only_episodes for p in data['perception'][e]}
    shared_episode_images = {p for e in shared_episodes for p in data['perception'][e]}
    from PIL import Image, ImageChops, ImageStat, ImageOps
    pixel_differences, dimensions, examples = [], Counter(), []
    for episode in sorted(shared_episodes):
        first_policy, first_perception = data['policy'][episode][0], data['perception'][episode][0]
        with Image.open(source/'images'/first_policy) as pa, Image.open(source/'images'/first_perception) as pb:
            a,b = ImageOps.exif_transpose(pa).convert('RGB'),ImageOps.exif_transpose(pb).convert('RGB')
            dimensions[f'{a.width}x{a.height} <- {b.width}x{b.height}'] += 1
            difference = sum(ImageStat.Stat(ImageChops.difference(a.resize((64,64)),b.resize((64,64)))).mean)/3
            pixel_differences.append(difference)
            if len(examples)<5:
                examples.append({'episode':episode,'policy_image':first_policy,'perception_image':first_perception,
                                 'policy_size':a.size,'perception_size':b.size,'resized_mean_abs_difference_0_255':difference})
    report = {'delivery_files':len(delivery),'policy_unique_files':len(policy),'perception_unique_files':len(perception),
        'shared_files':len(policy & perception),'delivery_not_policy':len(delivery-policy),
        'delivery_unreferenced_by_either':len(delivery-(policy|perception)),
        'shared_episodes':len(shared_episodes),'perception_only_episodes':len(legacy_only_episodes),
        'nonpolicy_files_from_shared_episodes':len((perception-policy)&shared_episode_images),
        'nonpolicy_files_from_legacy_only_episodes':len((perception-policy)&legacy_only_images),
        'shared_episodes_identical_first_image_filename':sum(data['policy'][e][0]==data['perception'][e][0] for e in shared_episodes),
        'first_image_pixel_difference_after_resize':dist(pixel_differences),
        'first_image_dimension_pairs_top10':dimensions.most_common(10),'examples':examples,
        'note':'Pixel distance is diagnostic, not proof of semantic equivalence. No data/images changed.'}
    write(args.output, report)
    import json
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
