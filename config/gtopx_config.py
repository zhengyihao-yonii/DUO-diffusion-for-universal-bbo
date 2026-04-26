import os
import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # misc（SOO-Bench GTOPX：默认 gtopx2，可用 eval_task / dataset 覆盖为 gtopx3,4,6）
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = 'trained_models/'
    dataset = 'gtopx2'
    fixed_dim = 128

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianDiffusion'
    horizon = 64
    n_diffusion_steps = 200
    train_timestep_bias_power = 0.0
    train_loss_min_snr_gamma = 0.0
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    # 显式「标量 return」条件（returns_mlp）；默认 False。默认 GTG 式目标值在轨迹最后一维
    # （见 construct / PointRegretDataset，y 已按任务归一化到 [0,1]），不经此开关。
    returns_condition = False
    calc_energy = False
    dim = 128
    condition_dropout = 0.25
    condition_guidance_w = 1.2
    # Task × text 联合 CFG（采样）；0 表示该轴不参与 CFG（与 returns_condition 的 guidance 独立）
    condition_guidance_w_task = 0.0
    condition_guidance_w_text = 0.0
    cfg_apply_task = True
    cfg_apply_text = True
    sample_with_task_embedding = True
    sample_with_text_embedding = True
    test_ret = 0.9
    renderer = None

    ## dataset
    loader = 'datasets.PointRegretDataset'
    proxy_loader = 'datasets.ZipDataset'
    data_path = 'generated_datasets/'
    context_length = 32
    regret = False

    clip_denoised = True
    # 为 True 且 returns_condition=True 时，训练用 RewardBatch 把标量 return 传入扩散损失
    include_returns = False
    train_only_inv = False

    ## training
    n_steps_per_epoch = 100
    loss_type = 'l2'
    n_train_steps = 50000
    batch_size = 128
    learning_rate = 1e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 50
    save_freq = 5000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = True

    ## proxy_model
    proxy_model = "models.Proxy"
    proxy_hidden_dim = 1024
    proxy_n_ensembles = 10
    proxy_learning_rate = 1e-3
    proxy_n_train_steps = 5000
    proxy_log_freq = 100
    proxy_save_freq = 1000
    proxy_filter = True
