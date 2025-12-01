import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os 
import pickle as pkl
import gym
# from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.utils.des_bench import DesignBenchFunctionWrapper
from diffuser.datasets.d4rl import suppress_output
from diffuser.models.vae import VAE
# 添加StandardScaler导入
from sklearn.preprocessing import StandardScaler


def evaluate(**deps):
    from ml_logger import logger, RUN

    RUN._update(deps)
    print(deps)
    if deps['task'] == 'ant':
        from config.ant_config import Config
    elif deps['task'] == 'dkitty':
        from config.dkitty_config import Config
    elif deps['task'] == 'tfbind8':
        from config.tfbind8_config import Config
    elif deps['task'] == 'tfbind10':
        from config.tfbind10_config import Config
    elif deps['task'] == 'superconductor':
        from config.superconductor_config import Config
    Config._update(deps)
    
    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'
    
    loadpath = os.path.join(logger.prefix, 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{Config.n_train_steps}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=Config.device)
    
    proxy_loadpath = os.path.join(logger.prefix, 'proxy_checkpoint')
    
    if Config.save_checkpoints:
        proxy_loadpath = os.path.join(proxy_loadpath, f'state_{Config.proxy_n_train_steps}.pt')
    else:
        proxy_loadpath = os.path.join(proxy_loadpath, 'state.pt')
    
    proxy_state_dict = torch.load(proxy_loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        # env=Config.dataset,
        horizon=Config.horizon,
        data_path=Config.data_path,
        context_length=Config.context_length,
        regret=Config.regret,
        # normalizer=Config.normalizer,
        # preprocess_fns=Config.preprocess_fns,
        # use_padding=Config.use_padding,
        # max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        # returns_scale=Config.returns_scale,
    )

    proxy_dataset_config = utils.Config(
        Config.proxy_loader,
        dataset=Config.dataset,
        frac=Config.frac,
        sigma=Config.sigma,
        savepath='proxy_dataset_config.pkl',
    )

    # render_config = utils.Config(
    #     Config.renderer,
    #     savepath='render_config.pkl',
    #     env=Config.dataset,
    # )

    dataset = dataset_config()
    proxy_dataset = proxy_dataset_config()
    # renderer = render_config()
    renderer = Config.renderer
    observation_dim = dataset.observation_dim
    original_observation_dim = proxy_dataset.original_observation_dim
    action_dim = dataset.action_dim
    print(observation_dim, action_dim)
    
    # 加载VAE模型用于从隐空间解码到原始空间
    vae = None
    # original_observation_dim = observation_dim  # 默认使用当前的观测维度
    latent_dim = observation_dim  # 隐空间维度，默认与当前观测维度相同
    vae_input_output_dim = 128  # 根据用户设计，VAE的输入输出维度固定为128
    
    # 尝试加载VAE信息
    vae_info_path = f"./generated_datasets/{Config.task}_frac{Config.frac}_sigma{Config.sigma}/vae_info.p"
    if os.path.exists(vae_info_path):
        try:
            with open(vae_info_path, 'rb') as f:
                vae_info = pkl.load(f)
            
            # 获取VAE路径
            vae_path = vae_info.get('vae_path')
            latent_dim = vae_info.get('latent_dim', observation_dim)
            
            print(f"从zipdataset加载原始观测维度: {original_observation_dim}")
            print(f"从VAE信息中加载VAE路径: {vae_path}")
            print(f"从VAE信息中加载隐空间维度: {latent_dim}")
            print(f"使用固定的VAE输入/输出维度: {vae_input_output_dim}")
            
            # 加载VAE模型 - 使用固定的128维输入/输出
            vae = VAE(input_dim=vae_input_output_dim, latent_dim=latent_dim)
            vae.load_state_dict(torch.load(vae_path, map_location=Config.device))
            vae.to(Config.device)
            vae.eval()
            print(f"VAE模型加载成功")
            
            # 加载VAE训练时使用的scaler参数
            scaler = StandardScaler()
            model_save_dir = os.path.dirname(vae_path)
            scaler_mean_path = os.path.join(model_save_dir, "scaler_mean.npy")
            scaler_scale_path = os.path.join(model_save_dir, "scaler_scale.npy")
            
            if os.path.exists(scaler_mean_path) and os.path.exists(scaler_scale_path):
                scaler.mean_ = np.load(scaler_mean_path)
                scaler.scale_ = np.load(scaler_scale_path)
                scaler.n_features_in_ = len(scaler.mean_)
                print(f"成功加载scaler参数，均值形状: {scaler.mean_.shape}")
            else:
                print(f"警告: 无法找到scaler参数文件，将不进行scaler还原")
                scaler = None

        except Exception as e:
            print(f"加载VAE模型时出错: {e}")
            vae = None

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )
    
    # 更新proxy_model_config的input_dim以匹配原始观测维度
    proxy_model_config = utils.Config(
        Config.proxy_model,
        savepath='proxy_model_config.pkl',
        input_dim=original_observation_dim,  # 使用原始观测维度
        hidden_dim=Config.proxy_hidden_dim,
        output_dim=action_dim,
        n_ensembles=Config.proxy_n_ensembles,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        # hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
    )
    
    Config.batch_size = 128
    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        proxy_train_lr=Config.proxy_learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        proxy_save_freq=Config.proxy_save_freq,
        log_freq=Config.log_freq,
        proxy_log_freq=Config.proxy_log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    proxy_model = proxy_model_config()
    diffusion = diffusion_config(model)
    
    trainer = trainer_config(diffusion, proxy_model, dataset, proxy_dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    trainer.proxy_step = proxy_state_dict['step']
    trainer.proxy_model.load_state_dict(proxy_state_dict['model'])
    
    device = Config.device
    context_length = Config.ctx_len
    
    num_queries = 128
    num_eval = 1
    
    contexts = []
    queries = []
    for e in range(num_eval):        
        batch = next(trainer.dataloader)
        
        # context conditioning
        conditions = {i: to_torch(batch.trajectories[:, i+Config.horizon-context_length], device=device) for i in range(context_length)}
        conditions["ctx_len"] = to_torch(np.ones(trainer.batch_size,), device=device) * context_length
        
        # classifier-free guidance
        returns = torch.ones(1, ).to(device=device).unsqueeze(0) * Config.alpha
        returns = returns.repeat(trainer.batch_size, 1)
        
        samples, time = trainer.ema_model.conditional_sample(conditions, values=None, returns=returns)
        samples = samples[..., :observation_dim]
        print(f"生成的隐空间样本形状: {samples.shape}")
        
        # 如果VAE存在，将隐空间样本解码到原始空间
        if vae is not None:
            with torch.no_grad():
                # 确保samples和normalizer参数在同一个设备上
                # 将samples移至与normalizer参数相同的设备
                samples_device = samples.device
                # 将normalizer的maxs和mins移至与samples相同的设备
                if hasattr(trainer.dataset.normalizer, 'maxs'):
                    trainer.dataset.normalizer.maxs = trainer.dataset.normalizer.maxs.to(samples_device)
                    trainer.dataset.normalizer.mins = trainer.dataset.normalizer.mins.to(samples_device)
                # 对隐空间样本进行unnormalize处理
                unnormalized_samples = trainer.dataset.normalizer.unnormalize(samples)
                print(f"unnormalize后的隐空间样本形状: {unnormalized_samples.shape}")
                
                batch_size, horizon, _ = unnormalized_samples.shape
                print(f"原始隐空间样本形状: {unnormalized_samples.shape}")
                
                # 重塑为[batch_size*horizon, latent_dim]以适应VAE.decode的输入要求
                samples_flat = unnormalized_samples.reshape(-1, latent_dim)
                print(f"扁平化后的样本形状: {samples_flat.shape}")
                
                # 解码到128维空间 - 在二维进行
                decoded_flat = vae.decode(samples_flat)
                print(f"解码后的扁平化形状: {decoded_flat.shape}")
                
                # 在二维状态下截断前original_observation_dim维
                decoded_flat_truncated = decoded_flat[:, :original_observation_dim]
                print(f"截断后的扁平化形状: {decoded_flat_truncated.shape}")
                
                # 使用scaler进行反标准化 - 在二维状态下处理
                if scaler is not None:
                    print(f"应用scaler反标准化")
                    # 转换为numpy进行scaler处理
                    decoded_np = decoded_flat_truncated.cpu().numpy()
                    # 应用scaler.inverse_transform进行反标准化
                    decoded_inv_transformed = scaler.inverse_transform(decoded_np)
                    # 转换回tensor
                    decoded_flat_truncated = torch.tensor(decoded_inv_transformed, 
                                                         dtype=decoded_flat_truncated.dtype, 
                                                         device=decoded_flat_truncated.device)
                    print(f"scaler反标准化后的样本范围: min={decoded_inv_transformed.min()}, max={decoded_inv_transformed.max()}")
                
                # 最后重塑回三维[batch_size, horizon, original_observation_dim]
                decoded_samples = decoded_flat_truncated.reshape(batch_size, horizon, original_observation_dim)
                print(f"重塑后的三维解码样本形状: {decoded_samples.shape}")
                
                
                samples = decoded_samples
            print(f"最终样本形状: {samples.shape}")

        queries.append(samples[:, context_length:])
        contexts.append(samples[:, :context_length])

    # 确保使用正确的维度进行reshape
    queries = torch.cat(queries, dim=0).reshape(-1, original_observation_dim if vae is not None else observation_dim)
    contexts = torch.cat(contexts, dim=0).reshape(-1, original_observation_dim if vae is not None else observation_dim).cpu().numpy()
    print(queries.shape, contexts.shape)
    
    # 对于解码后的样本，需要调整归一化处理以匹配原始观测空间
    queries_cpu = queries.cpu()
    
    # 如果使用了VAE解码，我们需要确保归一化处理正确
    if vae is not None:
            # VAE解码后的样本可能不在原始空间范围内，需要适当处理
            # 首先将样本转换为numpy数组
            queries_np = queries_cpu.numpy()
            # 记录解码后样本的范围
            print(f"解码后样本范围: min={queries_np.min()}, max={queries_np.max()}")
            # 使用proxy_dataset的normalizer对解码后的样本进行归一化
            # 由于在VAE解码前已经进行了unnormalize，这里直接使用queries_cpu
            queries_unnorm_tensor = queries_cpu.to('cpu')
            # 对于VAE解码后的样本，我们直接进行归一化
            queries_norm = trainer.proxy_dataset.normalizer.normalize(queries_unnorm_tensor)
    else:
        # 没有VAE时，使用原始的归一化处理
        queries_unnorm = trainer.dataset.normalizer.unnormalize(queries_cpu)
        queries_unnorm_tensor = torch.tensor(queries_unnorm, device='cpu')
        queries_norm = trainer.proxy_dataset.normalizer.normalize(queries_unnorm_tensor)
    
    # 归一化后再转移到训练设备
    queries_norm = queries_norm.to(trainer.device)
    queries_proxy_score = trainer.proxy_model(queries_norm).flatten()

    # filtering
    queries = queries[torch.argsort(queries_proxy_score)[-num_queries:]].cpu()
    
    # 如果使用了VAE，queries已经在原始空间中，不需要额外的unnormalize操作
    if vae is None:
        queries = dataset.normalizer.unnormalize(queries).numpy()
    else:
        # 已经在原始空间中的样本直接转换为numpy
        queries = queries.numpy()
            
    func = DesignBenchFunctionWrapper(deps["task"], normalise=True)
    if deps["task"].startswith("tfbind"):
        queries = func.task.to_integers(queries.reshape(num_queries, -1, 3))
    else:
        queries = queries.reshape(num_queries, -1)
    y = func.task.predict(queries)
    y_norm = (y - func.min) / (func.max - func.min)
    
    logger.print(f"max_ep_reward: {np.max(y)}, median_ep_reward: {np.median(y)}, mean_ep_reward: {np.mean(y)},", color='green')
    logger.log_metrics_summary({f"max_ep_reward": np.max(y), "median_ep_reward": np.median(y), "mean_ep_reward": np.mean(y)})
    
    logger.print(f"nmax_ep_reward: {np.max(y_norm)}, nmedian_ep_reward: {np.median(y_norm)}, nmean_ep_reward: {np.mean(y_norm)},", color='green')
    logger.log_metrics_summary({f"nmax_ep_reward": np.max(y_norm), "nmedian_ep_reward": np.median(y_norm), "nmean_ep_reward": np.mean(y_norm)})
    
    np.savez_compressed(os.path.join(logger.prefix, f'performance_{Config.n_train_steps}_{trainer.batch_size}x{Config.horizon - context_length}_alpha{Config.alpha}'), y=y, y_norm=y_norm, time=time)
    np.savez_compressed(os.path.join(logger.prefix, f'samples_{Config.n_train_steps}_{trainer.batch_size}x{Config.horizon - context_length}_alpha{Config.alpha}'), queries=queries)
