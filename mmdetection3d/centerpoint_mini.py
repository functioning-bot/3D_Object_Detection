_base_ = 'checkpoints/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py'

# Override to use v1.0-mini
test_dataloader = dict(
    dataset=dict(
    ann_file='nuscenes_infos_val.pkl',
    )
)

test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    data_root='data/nuscenes/',
    metric='bbox',
    jsonfile_prefix='results/nuscenes_mini',
)

# This is the key fix - tell NuScenes to use mini version
_base_.test_evaluator.update(dict(version='v1.0-mini'))