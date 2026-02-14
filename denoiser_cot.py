import torch
import torch.nn as nn
from model_cot import JiTCoT_models
from dinov2_hf import RAE
from mocov3 import MocoV3
from data2vec2 import Data2Vec2Encoder
from pixel_downsample import PixelDownsampleEncoder

class DenoiserCoT(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiTCoT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            bottleneck_dim_dino=args.bottleneck_dim_dino,
            dh_depth=args.dh_depth,
            dh_hidden_size=args.dh_hidden_size,
            dino_in_channels=args.dino_in_channels,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.D_mean = args.D_mean
        self.D_std = args.D_std
        self.t_eps = args.t_eps
        self.t_eps_inference = args.t_eps_inference
        self.noise_scale = args.noise_scale

        if args.latent_model == "dino":
            self.dino = RAE()
            self.dino_hidden_dim = 768
        elif args.latent_model == "mocov3":
            self.dino = MocoV3()
            self.dino_hidden_dim = 768
        elif args.latent_model == "d2v2":
            self.dino = Data2Vec2Encoder()
            self.dino_hidden_dim = 1024
        elif args.latent_model == "pixeldownsample":
            self.dino = PixelDownsampleEncoder()
            self.dino_hidden_dim = 48
        else:
            raise NotImplementedError()


        self.dino_max_t = args.dino_max_t
        self.dino_weight = args.dino_weight
        self.sample_mode = args.sample_mode
        self.choose_dino_p = args.choose_dino_p
        self.dino_pixel_offset = args.dino_pixel_offset
        self.dino_pixel_shift = args.dino_pixel_shift

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_scale_dino = args.cfg_dino
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.cfg_interval_dino = (args.interval_min_dino, args.interval_max_dino)
        self.gen_shift_pixel = args.gen_shift_pixel
        self.gen_shift_dino = args.gen_shift_dino
        self.guidance_method = args.guidance_method

        self.mask_p = args.mask_p

        self.ag_net = None
        if args.autoguidance_ckpt != "":
            self.ag_net = JiTCoT_models['JiTCoT-S/16'](
                input_size=args.img_size,
                in_channels=3,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
                bottleneck_dim_dino=args.bottleneck_dim_dino,
                dh_depth=2,
                dh_hidden_size=1024,
            )
            self.ag_net.requires_grad_(False)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    @torch.autocast('cuda', enabled=False)
    def sample_t(self, n: int, device=None, dtype=torch.bfloat16):
        if self.sample_mode in {"shifted_independent_uniform", "shifted_independent_uniform_masked"}:
            ts = 0.5
            t = torch.rand((2, n), device=device)
            t = t * ts / (1 + (ts-1) * t)
            loss_weight = [torch.ones(n, device=device), torch.ones(n, device=device)]
            return [t[0], t[1]], loss_weight
        elif self.sample_mode == "dino_only":
            t_dino = torch.sigmoid(torch.randn(n, device=device) * self.D_std + self.D_mean)
            t_pixel = torch.zeros(n, device=device)

            t = [t_pixel.to(dtype), t_dino.to(dtype)]
            loss_weight = [torch.zeros(n, device=device, dtype=dtype), torch.ones(n, device=device, dtype=dtype)]
            return t, loss_weight
        elif self.sample_mode == "pixel_only":
            t_pixel = torch.sigmoid(torch.randn(n, device=device) * self.P_std + self.P_mean)
            t_dino = torch.zeros(n, device=device)

            t = [t_pixel.to(dtype), t_dino.to(dtype)]
            loss_weight = [torch.ones(n, device=device, dtype=dtype), torch.zeros(n, device=device, dtype=dtype)]
            return t, loss_weight
        elif self.sample_mode == "dino_first_cascaded":
            t_dino = torch.sigmoid(torch.randn(n, device=device) * self.D_std + self.D_mean)
            t_pixel = torch.sigmoid(torch.randn(n, device=device) * self.P_std + self.P_mean)
            t_pixel = torch.where(torch.rand(n, device=device) < 0.1, torch.rand(n, device=device) * 0.5, t_pixel) # nonzero t=0

            choose_dino_mask = torch.rand(n, device=device) < self.choose_dino_p
            t_dino = torch.where(choose_dino_mask, t_dino, torch.ones_like(t_dino))
            t_pixel = torch.where(choose_dino_mask, torch.zeros_like(t_pixel), t_pixel)

            t = [t_pixel.to(dtype), t_dino.to(dtype)]
            loss_weight = [1.0*(~choose_dino_mask), 1.0*choose_dino_mask]
            return t, loss_weight
        elif self.sample_mode == "dino_first_cascaded_noised":
            t_dino = torch.sigmoid(torch.randn(n, device=device) * self.D_std + self.D_mean)
            t_pixel = torch.sigmoid(torch.randn(n, device=device) * self.P_std + self.P_mean)
            t_pixel = torch.where(torch.rand(n, device=device) < 0.1, torch.rand(n, device=device) * 0.5, t_pixel) # nonzero t=0

            choose_dino_mask = torch.rand(n, device=device) < self.choose_dino_p
            t_dino = torch.where(choose_dino_mask, t_dino, torch.rand_like(t_dino) * 0.25 + 0.75) # Noise beta = 0.25
            t_pixel = torch.where(choose_dino_mask, torch.zeros_like(t_pixel), t_pixel)

            t = [t_pixel.to(dtype), t_dino.to(dtype)]
            loss_weight = [1.0*(~choose_dino_mask), 1.0*choose_dino_mask]
            return t, loss_weight
        elif self.sample_mode == "dino_first_shifted_aligned":
            t_dino = torch.sigmoid(torch.randn(n, device=device) * self.D_std + self.D_mean)
            t_pixel = torch.sigmoid(torch.randn(n, device=device) * self.P_std + self.P_mean)

            def local_shift(t,a):
                return t * a / (1 + (a-1)*t)

            choose_dino_mask = torch.rand(n, device=device) < self.choose_dino_p
            t_dino = torch.where(choose_dino_mask, t_dino, local_shift(t_pixel, self.dino_pixel_shift))
            t_pixel = torch.where(choose_dino_mask, local_shift(t_dino, 1/self.dino_pixel_shift), t_pixel)

            t = [t_pixel.to(dtype), t_dino.to(dtype)]
            loss_weight = [1.0*(~choose_dino_mask), 1.0*choose_dino_mask] # There's a lower variance way to do this
            return t, loss_weight
        
    def get_generate_timesteps(self, labels, z):
        device = labels.device
        bsz = labels.size(0)
        if self.sample_mode in {"eval_shifted", "dino_first_shifted_aligned", "shifted_independent_uniform"}:
            # Balances the diffusion steps along the curved trajectory, inverts this to get dino/pixel timesteps
            # Input using shift**0.5. This code creates a variance shift of $alpha$ by multipliying and dividing by sqrt(alpha) for pixel and dino
            def get_schedule(n, off, s, dev):
                y = torch.linspace(0, 1, n + 1, device=dev)
                t_d = y / (s - (s - 1) * y)
                t_p = y / ((1/s) - ((1/s) - 1) * y)
                
                t_d = t_d * (1 + off) - off if off < 0 else t_d
                t_p = t_p * (1 - off) + off if off >= 0 else t_p
                
                t = torch.sort(torch.cat([t_d, t_p]))[0][::2]

                def warp(v, p, o, dino):
                    dly = (o < 0 and dino) or (o > 0 and not dino)
                    vn = (v - abs(o)).clamp(0) / (1 - abs(o)) if dly else v
                    return (p * vn) / (1 + (p - 1) * vn)

                return torch.stack([warp(ti, 1/s, off, False) for ti in t]), \
                    torch.stack([warp(ti, s, off, True) for ti in t])

            tp, td = get_schedule(self.steps, self.dino_pixel_offset, self.dino_pixel_shift**0.5 if self.sample_mode == "dino_first_shifted_aligned" else self.dino_pixel_shift, device)
            
            timesteps_pixel = tp.view(-1, *([1] * z[0].ndim)).expand(-1, bsz, -1, -1, -1)
            timesteps_dino = td.view(-1, *([1] * z[1].ndim)).expand(-1, bsz, -1, -1, -1)
            timesteps = list(zip(timesteps_pixel, timesteps_dino))
        elif self.sample_mode in {"dino_first_cascaded", "dino_first_cascaded_noised"}:
            timesteps_pixel = torch.cat([torch.zeros(self.steps//2+1, device=device), torch.linspace(0.0, 1.0, self.steps//2+1, device=device)[1:]], dim=0).view(-1, *([1] * z[1].ndim)).expand(-1, bsz, -1, -1, -1)
            timesteps_dino = torch.cat([torch.linspace(0.0, self.dino_max_t, self.steps//2+1, device=device), torch.full((self.steps//2+1,), self.dino_max_t, device=device)[1:]], dim=0).view(-1, *([1] * z[0].ndim)).expand(-1, bsz, -1, -1, -1)

            timesteps_pixel = timesteps_pixel * self.gen_shift_pixel / (1 + (self.gen_shift_pixel - 1) * timesteps_pixel)
            timesteps_dino = timesteps_dino * self.gen_shift_dino / (1 + (self.gen_shift_dino - 1) * timesteps_dino)
            timesteps = list(zip(timesteps_pixel, timesteps_dino))
        return timesteps

    def forward(self, x, labels):
        # x: [-1,1] min,max; Image-shape N 3 H W
        labels_dropped = self.drop_labels(labels) if self.training else labels

        bsz, device = x.size(0), x.device
        x_pixel = x
        x_dino = self.dino.encode(x * 0.5 + 0.5)
        x = [x_pixel, x_dino]
        t, loss_weight = self.sample_t(bsz, device=device)
        t = [t[i].view(-1, *([1] * (x[i].ndim - 1))) for i in [0,1]]
        e = [torch.randn_like(x[i]) * self.noise_scale for i in [0,1]]

        z = [t[i] * x[i] + (1 - t[i]) * e[i] for i in [0,1]]
        v = [(x[i] - z[i]) / (1 - t[i]).clamp_min(self.t_eps) for i in [0,1]]

        x_pred = self.net(z, [t[i].flatten() for i in [0,1]], labels_dropped, m=m)
        v_pred = [(x_pred[i] - z[i]) / (1 - t[i]).clamp_min(self.t_eps) for i in [0,1]]

        # l2 loss
        losses = [((v[i] - v_pred[i]).pow(2).mean(dim=(1, 2, 3)) * loss_weight[i]).mean() for i in [0,1]]
        loss = losses[0] + losses[1] * self.dino_weight
        results_dict = {'loss_pixel': losses[0].item(), 'loss_dino': losses[1].item()}

        return loss, results_dict

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z_pixel = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        z_dino = self.noise_scale * torch.randn(bsz, self.dino_hidden_dim, 16, 16, device=device)
        z = [z_pixel, z_dino]
        timesteps = self.get_generate_timesteps(labels, z)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        q = None
        for i in range(self.steps - 1):
            if q is None and timesteps[i][1].view(-1)[0].item() >= 0.75:
                q = (z[1], timesteps[i][1])
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels, q if i < self.steps // 2 else q)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, q)
        z_pixel, z_dino = z
        return z_pixel

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, q=None):
        # conditional
        x_cond = self.net(z, [t[i].flatten() for i in [0,1]], labels)
        v_cond = [(x_cond[i] - z[i]) / (1.0 - t[i]).clamp_min(self.t_eps_inference) for i in [0,1]]

        # unconditional
        if self.guidance_method == "autoguidance":
            x_uncond = self.ag_net(z, [t[i].flatten() for i in [0,1]], torch.full_like(labels, self.num_classes))
        elif self.guidance_method == "cfg":
            x_uncond = self.net(z, [t[i].flatten() for i in [0,1]], torch.full_like(labels, self.num_classes))
        elif self.guidance_method == "cfg_interval":
            if self.ag_net is not None and t[1].flatten()[1].item() >= 0.999:
                x_uncond = self.ag_net(z, [t[i].flatten() for i in [0,1]], torch.full_like(labels, self.num_classes))
            else:
                x_uncond = self.net(z, [t[i].flatten() for i in [0,1]], torch.full_like(labels, self.num_classes))

        v_uncond = [(x_uncond[i] - z[i]) / (1.0 - t[i]).clamp_min(self.t_eps_inference) for i in [0,1]]

        # cfg interval
        low, high = self.cfg_interval
        low_dino, high_dino = self.cfg_interval_dino
        interval_mask = [(t[0] < high) & ((low == 0) | (t[0] > low)), (t[1] < high_dino) & ((low_dino == 0) | (t[1] > low_dino))]
        cfg_scale_interval = [torch.where(interval_mask[i], [self.cfg_scale, self.cfg_scale_dino][i], 1.0) for i in [0,1]]

        return [v_uncond[i] + cfg_scale_interval[i] * (v_cond[i] - v_uncond[i]) for i in [0,1]]

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, q):
        v_pred = self._forward_sample(z, t, labels, q)
        z_next = [z[i] + (t_next[i] - t[i]) * v_pred[i] for i in [0,1]]
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, q=None):
        v_pred_t = self._forward_sample(z, t, labels, q)

        z_next_euler = [z[i] + (t_next[i] - t[i]) * v_pred_t[i] for i in [0,1]]
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, q)

        v_pred = [0.5 * (v_pred_t[i] + v_pred_t_next[i]) for i in [0,1]]
        z_next = [z[i] + (t_next[i] - t[i]) * v_pred[i] for i in [0,1]]
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
