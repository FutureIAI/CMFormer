import ml_collections

model = 'CMFormer-S'

if model == 'CMFormer-S':
    base_blocks = [2, 2, 2, 2]
    ms_cac_blocks = [2, 2, 2, 2]
    decoder_channel = 128
    pretrain_weight_dir = r"./pretrain_weight/pvt_v2/pvt_v2_b1.pth"
elif model == 'CMFormer-M':
    base_blocks = [3, 4, 6, 3]
    ms_cac_blocks = [3, 4, 6, 3]
    decoder_channel = 256
    pretrain_weight_dir = r"./pretrain_weight/pvt_v2/pvt_v2_b2.pth"
elif model == 'CMFormer-L':
    base_blocks = [3, 4, 18, 3]
    ms_cac_blocks = [3, 4, 12, 3]
    decoder_channel = 512
    pretrain_weight_dir = r"./pretrain_weight/pvt_v2/pvt_v2_b3.pth"


def get_config():
    config = ml_collections.ConfigDict()
    config.image_h = 480
    config.image_w = 640
    config.classes = 40
    config.num_stages = 4
    config.train_file = r"./datasets/nyuv2/train.txt"
    config.val_file = r"./datasets/nyuv2/test.txt"
    config.pretrain_weight_dir = pretrain_weight_dir
    config.train_batch_size = 4
    config.val_batch_size = 1
    config.num_workers = 4
    config.begin_epoch = 0
    config.stop_epoch = 300
    config.save_freq = 10

    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.lr = 0.00006
    config.optimizer.wd = 0.01
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.power = 0.9
    config.lr_scheduler.warm_up_epoch = 5

    # TokenEmbed config
    config.embed = ml_collections.ConfigDict()
    config.embed.channels = [3, 64, 128, 320, 512]
    config.embed.kernel_size = [7, 3, 3, 3]
    config.embed.stride = [4, 2, 2, 2]
    config.embed.padding = [3, 1, 1, 1]

    # Transformer Block config
    config.trans = ml_collections.ConfigDict()
    config.trans.dims = [64, 128, 320, 512]
    config.trans.blocks = base_blocks
    config.trans.drop_path = 0.1

    # attn
    config.trans.attn = ml_collections.ConfigDict()
    config.trans.attn.dims = [64, 128, 320, 512]
    config.trans.attn.sr_ratios = [8, 4, 2, 1]
    config.trans.attn.num_heads = [1, 2, 5, 8]
    config.trans.attn.qkv_bias = True
    config.trans.attn.attn_drop = 0.0
    config.trans.attn.proj_drop = 0.0
    # mlp
    config.trans.mlp = ml_collections.ConfigDict()
    config.trans.mlp.in_features = [64, 128, 320, 512]
    config.trans.mlp.mlp_ratios = [8, 8, 4, 4]
    config.trans.mlp.hidden_features = [a * b for a, b in zip(config.trans.attn.dims, config.trans.mlp.mlp_ratios)]
    config.trans.mlp.drop = 0.0

    # MS_CAC
    config.MS_CAC = ml_collections.ConfigDict()
    config.MS_CAC.I2V = ml_collections.ConfigDict()
    config.MS_CAC.I2V.dims = [64, 128, 320, 512]
    config.MS_CAC.I2V.target_size = [(1, 1), (3, 4), (6, 8), (12, 16)]
    config.MS_CAC.I2V.spp_dims = sum([x * y for x, y in config.MS_CAC.I2V.target_size])
    config.MS_CAC.I2V.out_dims = [512, 320, 128, 64]

    config.MS_CAC.trans = ml_collections.ConfigDict()
    config.MS_CAC.trans.dims = [512, 320, 128, 64]
    config.MS_CAC.trans.blocks = ms_cac_blocks
    config.MS_CAC.trans.drop_path = 0.1

    # attn
    config.MS_CAC.trans.attn = ml_collections.ConfigDict()
    config.MS_CAC.trans.attn.dims = [512, 320, 128, 64]
    config.MS_CAC.trans.attn.num_heads = [8, 5, 2, 1]
    config.MS_CAC.trans.attn.qkv_bias = True
    config.MS_CAC.trans.attn.attn_drop = 0.0
    config.MS_CAC.trans.attn.proj_drop = 0.0
    # mlp
    config.MS_CAC.trans.mlp = ml_collections.ConfigDict()
    config.MS_CAC.trans.mlp.in_features = [512, 320, 128, 64]
    config.MS_CAC.trans.mlp.mlp_ratios = [4, 4, 8, 8]
    config.MS_CAC.trans.mlp.hidden_features = [a * b for a, b in zip(config.MS_CAC.trans.attn.dims,
                                                                     config.MS_CAC.trans.mlp.mlp_ratios)]
    config.MS_CAC.trans.mlp.drop = 0.0

    # GFA
    config.GFA = ml_collections.ConfigDict()
    config.GFA.trans = ml_collections.ConfigDict()
    config.GFA.trans.dims = [64, 128, 320, 512]
    config.GFA.trans.drop_path = 0.1

    # attn
    config.GFA.trans.attn = ml_collections.ConfigDict()
    config.GFA.trans.attn.dims = [64, 128, 320, 512]
    config.GFA.trans.attn.sr_ratios = [8, 4, 2, 1]
    config.GFA.trans.attn.num_heads = [1, 2, 5, 8]
    config.GFA.trans.attn.qkv_bias = True
    config.GFA.trans.attn.attn_drop = 0.0
    config.GFA.trans.attn.proj_drop = 0.0
    # mlp
    config.GFA.trans.mlp = ml_collections.ConfigDict()
    config.GFA.trans.mlp.in_features = [64, 128, 320, 512]
    config.GFA.trans.mlp.mlp_ratios = [4, 4, 2, 2]
    config.GFA.trans.mlp.hidden_features = [a * b for a, b in zip(config.GFA.trans.attn.dims,
                                                                  config.GFA.trans.mlp.mlp_ratios)]
    config.GFA.trans.mlp.drop = 0.0

    # head
    config.SegFormer_head = ml_collections.ConfigDict()
    config.SegFormer_head.embedding_dim = decoder_channel
    config.SegFormer_head.drop_out = 0.0

    config.fpn_head = ml_collections.ConfigDict()
    config.fpn_head.dims = config.trans.dims
    config.fpn_head.channels = decoder_channel
    config.fpn_head.dropout_ratio = 0.1
    config.fpn_head.in_index = [0, 1, 2, 3]
    config.fpn_head.feature_strides = [4, 8, 16, 32]

    return config
