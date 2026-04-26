import os
import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # misc
    seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = 'trained_models/'
    dataset = 'ant'
    # 多任务时若首任务为 ant（如 "ant,dkitty"），实验走本文件而非 multi_dataset_config（后者未被 train/evaluate 导入）
    fixed_dim = 128

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianDiffusion'
    horizon = 64
    n_diffusion_steps = 200
    n_sample_timesteps = 200
    # 可选扩散训练改进（默认 0；CLI --train_timestep_bias_power / --train_loss_min_snr_gamma 可覆盖）
    train_timestep_bias_power = 0.0
    train_loss_min_snr_gamma = 0.0
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = False
    calc_energy = False
    dim = 128
    condition_dropout = 0.25
    # 多任务仅文本分支（与 multitask_text_only CLI 一致）；默认 False = 多任务仍用 task 分类条件
    multitask_text_only = False
    # Few-shot：单任务轨迹上仅文本条件微调（与 multitask_text_only 同：无 task 分支）
    fewshot_text_only_finetune = False  # 兼容旧名；请用 real_task_text_only_finetune
    real_task_text_only_finetune = False
    load_diffusion_checkpoint = None
    load_diffusion_checkpoint_epoch = None
    # 任务描述句向量（与 task one-hot 相加）；见 task_metadata/README.md
    use_text_condition = False
    task_metadata_dir = 'task_metadata'
    text_encoder_model = 'sentence-transformers/all-MiniLM-L6-v2'
    text_condition_dropout = 0.1
    condition_guidance_w = 1.2
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
    include_returns = False
    
    # normalizer = 'CDFNormalizer'
    # preprocess_fns = []
    clip_denoised = True
    # use_padding = True
    # include_returns = True
    # discount = 0.99
    # max_path_length = 1000
    # hidden_dim = 256
    # ar_inv = False
    train_only_inv = False
    # termination_penalty = -100
    # returns_scale = 400.0 # Determined using rewards from the dataset

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
    # 1=训练 proxy 且评估时用其筛选 queries；0=跳过（CLI/环境 PROXY_FILTER 可覆盖）
    proxy_filter = True
