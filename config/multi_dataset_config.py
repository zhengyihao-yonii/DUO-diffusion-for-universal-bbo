import os
import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # 注意：train/evaluate 不会 import 本文件；多任务实际使用「字典序首任务」的 config（如 ant,dkitty → ant_config）。
    # 本文件仅作参数模板或与 ParamsProto 手工合并时使用。
    # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = 'trained_models/'
    dataset = 'multi_tasks'  # 多任务标识符
    
    # 多任务特定参数
    tasks = ['tfbind8', 'tfbind10']  # 默认任务列表，可通过命令行覆盖
    fixed_dim = 128  # 固定的输入维度，用于统一不同数据集
    
    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianDiffusion'
    multitask_text_only = False
    fewshot_text_only_finetune = False
    real_task_text_only_finetune = False
    load_diffusion_checkpoint = None
    load_diffusion_checkpoint_epoch = None
    # Optional: task metadata text → sentence embedding (additive to task one-hot). See task_metadata/README.md
    use_text_condition = False
    task_metadata_dir = 'task_metadata'
    text_encoder_model = 'sentence-transformers/all-MiniLM-L6-v2'
    text_condition_dropout = 0.1
    horizon = 64
    n_diffusion_steps = 200
    train_timestep_bias_power = 0.0
    train_loss_min_snr_gamma = 0.0
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = False
    calc_energy = False
    dim = 32
    condition_dropout = 0.25
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
    
    # 多数据集统一参数，按照用户要求设置
    frac = 1.0
    sigma = 0.0
    n_traj = 1000  # 每任务生成的轨迹数量
    k = 20         # 每步选择的候选点数量
    eps = 0.05     # 允许的目标值下降范围
    
    clip_denoised = True
    include_returns = False
    train_only_inv = False
    
    # 训练参数
    n_steps_per_epoch = 100
    loss_type = 'l2'
    n_train_steps = 50000
    batch_size = 128
    learning_rate = 1e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 50
    save_freq = 3000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = True
    
    # VAE参数
    latent_dim = 32
    vae_d_model = 256
    vae_nhead = 4
    vae_num_layers = 4
    vae_dropout = 0.1
    vae_batch_size = 64
    vae_val_split = 0.1
    vae_lr = 1e-4
    vae_weight_decay = 1e-5
    vae_num_epochs = 100
    vae_kl_weight = 0.1
    
    # 代理模型参数
    proxy_model = "models.Proxy"
    proxy_hidden_dim = 1024
    proxy_n_ensembles = 10
    proxy_learning_rate = 1e-3
    proxy_n_train_steps = 3000
    proxy_log_freq = 100
    proxy_save_freq = 1000
    proxy_filter = True