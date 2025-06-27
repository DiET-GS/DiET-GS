import random
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionUpscalePipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import randn_tensor
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

import PIL
import numpy as np



@threestudio.register("stable-diffusion-upscaler-rsd-guidance")
class StableDiffusionUpscalerRSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-x4-upscaler"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        view_dependent_prompting: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        anneal_start_step: Optional[int] = 5000
        anneal_end_step: Optional[int] = 25000
        random_timestep: bool = True
        anneal_strategy: str = "milestone"
        max_step_percent_annealed: float = 0.5

        step_ratio: float = 0.25
        num_inference_steps: int = 20

        guidance_type: str = 'rsd'

        t_min_shift_per_stage: float = 0.0
        t_max_shift_per_stage: float = 0.0
        cfg_shift_per_stage: float = 0.0

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionUpscalePipeline

        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        self.submodules = SubModules(pipe=pipe)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # delete text encoders （text encoding is handled by prompt processor）
        del self.pipe.text_encoder
        cleanup()

        for p in self.vae.parameters(): # dont train pipe vae (pipe_lora also uses pipe vae, so vae is never trained)
            p.requires_grad_(False)
        for p in self.unet.parameters(): # dont train pipe unet
            p.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe.scheduler = self.scheduler

        self.low_res_scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="low_res_scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

        self.grad_clip_val: Optional[float] = None

        # Place unet onto gpu
        self.unet.to(self.device)
        threestudio.info(f"Loaded Stable Diffusion x4 Upscaler!")

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionUpscalePipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=class_labels,
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images

    def sample(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
        )


    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype

        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
        ).sample.to(input_dtype)
    

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)

        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)
        
        # print(f"Upcasted VAE from float16 to float32.")


    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 H/4 W/4"]:
        if self.vae.dtype==torch.float16:
            self.upcast_vae()

        input_dtype = imgs.dtype
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    # @torch.no_grad()
    def decode_latents(
        self, latents: Float[Tensor, "B 4 H/4 W/4"],
    ) -> Float[Tensor, "B 3 H W"]:
        if self.vae.dtype==torch.float16:
            self.upcast_vae()

        input_dtype = latents.dtype
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        # latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def set_custom_timesteps(
        self,
        scheduler : DPMSolverMultistepScheduler,
        last_timestep: Int,
        step_ratio: float,
        device: str,
        ):
        """
        Get timesteps for ancestral sampling from arbitrary t.
        timestep spacing follows trailing format.
        Args:
            scheduler ('DPMSolverMultistepScheduler'):
                scheduler for sample generation
            last_timestep ('float'):
                selected initial timestep for iterative denoising. Must be between (0.98,0.02)
            step_ratio ('float):
                timestep spacing given by round(num_train_timesteps*step_ratio)
        """

        # define step ratio in integers (number of DDPM steps per DPM step)
        step_ratio = round(step_ratio*scheduler.config.num_train_timesteps) # convert fractional step ratio into integer skips

        # define last_timestep in integers
        assert last_timestep < 1000 and last_timestep > 0, f"last_timestep out of accepted range: {last_timestep}"

        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        # clipped_idx = torch.searchsorted(torch.flip(scheduler.lambda_t, [0]), scheduler.config.lambda_min_clipped) #4
        # last_timestep = ((scheduler.config.num_train_timesteps - clipped_idx).numpy()).item() # we end at 996 instead of 1000

        # Use trailing format from given t
        timesteps = np.arange(last_timestep, 1, -step_ratio).round().copy().astype(np.int64)
        timesteps = np.append(timesteps,0)
        # timesteps -= 1

        sigmas = np.array(((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5)

        scheduler.sigmas = torch.from_numpy(sigmas)

        # when num_inference_steps == num_train_timesteps, we can end up with duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        scheduler.timesteps = torch.from_numpy(timesteps).to(device)

        scheduler.num_inference_steps = len(timesteps)

        scheduler.model_outputs = [None,] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0

        return scheduler
    

    # compute_grad_rsd() will process conditioning and noisify latents
    # then it passes noisy latents, t, image, noise_level, text_embeddings into iterative_denoise()
    # iterative_denoise() works like sample except given t and noisified latents
    # iterative_denoise() returns denoised_image
    # compute_grad_rsd() computes rsd loss with target-denoised_image

    @torch.cuda.amp.autocast(enabled=True)
    def compute_grad_rsd(
        self,
        latents: Float[Tensor, "B 4 256 256"],
        image: Float[Tensor, "B 3 256 256"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        noise_level: Float[Tensor, "BB"],
        ):

        B = latents.shape[0]

        # random timestamp
        if self.cfg.random_timestep:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
        else:
            t = torch.full([B], self.max_step, dtype=torch.long, device=self.device)
    
        last_timestep = t[0].detach().cpu().numpy()
        self.last_timestep = torch.tensor(last_timestep)

        sample_scheduler = self.set_custom_timesteps(self.scheduler_sample, last_timestep, self.cfg.step_ratio, device=self.device) # timesteps to apply unet (t+1)
        timesteps = sample_scheduler.timesteps

        discrete_idx = None
        prev_t = timesteps[1]

        # add noise to latents
        noise = torch.randn_like(latents)
        self.noise = noise

        latents_noisy = self.scheduler.add_noise(latents, noise, t) #[1,3,256,256] (zt) # to fit
        prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t) #[1,3,256,256] (zt-k) #input for unet

        # 1-step denoised image (SDS)
        with torch.no_grad():
            z_t, z_0 = self.onestep_denoise(pipe=self.pipe,
                                                    latents=latents_noisy, # zt+1 # test
                                                    image=image,
                                                    sample_scheduler=sample_scheduler,# [t+1]
                                                    text_embeddings=text_embeddings,
                                                    guidance_scale=self.cfg.guidance_scale,
                                                    noise_level=noise_level,
                                                    discrete_idx=discrete_idx,
                                                    )
            z_t = z_t.detach() # predicted zt from zt+1
            z_0 = z_0.detach() # predicted z0 from zt+1
            pred_latents_noisy = z_t

        return prev_latents_noisy, pred_latents_noisy


    def onestep_denoise(
        self,
        pipe: StableDiffusionUpscalePipeline,
        latents: Float[Tensor, "B 4 256 256"],
        image: Float[Tensor, "B 3 256 256"],
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise_level: int = 250,
        step_ratio: float = 0.1,
        discrete_idx: int = None
        ) -> Float[Tensor, "B H W 3"]:

        # 4. Prepare timesteps
        timesteps = sample_scheduler.timesteps
        
        if discrete_idx is not None:
            timesteps = timesteps[discrete_idx:]

        # disable this if we want ancestral sampling
        iterations = 1
        timesteps = timesteps[:iterations] # truncate timesteps by iterations required

        B = image.shape[0]
        noise_level = torch.cat([noise_level]*2)

        # 8. Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance

            latent_model_input = torch.cat([latents, image], dim=1)
            latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
            latent_model_input = sample_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipe.unet(
                    latent_model_input.to(self.weights_dtype),
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=noise_level,
            ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) #cfg  


            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample #xt-1 #BCHW

            # testing
            # if self.scheduler.config.prediction_type == 'v_prediction':
            #     noise_pred = self.velocity_to_epsilon(latent_model_input,noise_pred,self.scheduler,t)

            # # convert noise pred into x_0
            # latent_model_input = latent_model_input[:1,:4]
            # alpha_prod_t = sample_scheduler.alphas_cumprod[t]
            # beta_prod_t = 1 - alpha_prod_t
            # z_0 = (latent_model_input - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            # # renoise to x_t-1
            # if i < len(timesteps)-1:
            #     prev_t = timesteps[i+1]
            #     noise = torch.randn_like(latents)
            #     latents = self.scheduler.add_noise(z_0, noise, prev_t) #[1,3,256,256]

            # intermediate_latents = (latents / 2 + 0.5).clamp(0, 1) # cannot clamp (0,1). latents must be [-1,1]
            # intermediate_latents = intermediate_latents.permute(0, 2, 3, 1).float()
            # intermediate_latents = intermediate_latents[0].detach().cpu().numpy()
            # img_array.append(intermediate_latents)
        
        # visualize latents
    
        # x_0 = self.decode_latents(latents)    
        # x_0 = x_0.permute(0,2,3,1)[0].detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(x_0)
        # plt.show()
        # assert False

        # BGT loss (Test)
        # return latents.float()

        # SDS/RSD loss
        # convert v-prediction. 
        if self.scheduler.config.prediction_type == 'v_prediction':
            noise_pred = self.velocity_to_epsilon(latent_model_input,noise_pred,self.scheduler,t)

        # convert noise pred into z_0
        latent_model_input = latent_model_input[:B,:4]
        alpha_prod_t = sample_scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        z_0 = (latent_model_input - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        # visualization
        # print(f"timestep: {t}")
        # x_0 = self.decode_latents(z_0)    
        # x_0 = x_0.permute(0,2,3,1)[0].detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(x_0)
        # plt.show()
        # assert False

        z_0 = z_0.float()

        # return z_0

        return latents, z_0

    # If unet or unet_lora is trained on v-objective, convert its predictions to epsilon.
    def velocity_to_epsilon(self, latent_model_input, noise_pred, scheduler, t):

        assert scheduler.config.prediction_type == "v_prediction"

        B = noise_pred.shape[0]
    
        render_latents = latent_model_input[:B,:4] # [B,4,256,256] get latents w/o LR image

        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=render_latents.device, dtype=render_latents.dtype
        )
        alpha_t = alphas_cumprod[t] ** 0.5
        sigma_t = (1 - alphas_cumprod[t]) ** 0.5   

        noise_pred = render_latents * sigma_t + noise_pred * alpha_t

        return noise_pred


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 256 256"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        noise_level: Float[Tensor, "BB"],
    ):
        B = latents.shape[0]

        with torch.no_grad():
            # random timestamp
            if self.cfg.random_timestep:
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    [B],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                t = torch.full([B], self.max_step, dtype=torch.long, device=self.device)

            last_timestep = t[0].detach().cpu().numpy()
            self.last_timestep = torch.tensor(last_timestep)

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)


            # pred noise
            latent_model_input = torch.cat([latents_noisy, image], dim=1)
            latent_model_input = torch.cat([latent_model_input] * 2)
            noise_pred_pretrain = self.forward_unet(
                self.unet,
                latent_model_input.to(self.weights_dtype),
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                class_labels=noise_level,
            )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        # convert v-prediction. 
        if self.scheduler.config.prediction_type == 'v_prediction':
            noise_pred_pretrain = self.velocity_to_epsilon(latent_model_input,noise_pred_pretrain,self.scheduler,t)

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise) # epsilon SDS

        return grad

    def forward(
        self,
        render: Float[Tensor, "B H W C"],
        image: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput, 
        noise_level=20,
    ):  
        
        render = render.permute(0,3,1,2)
        image = image.permute(0,3,1,2)
        image = image * 2.0 - 1.0
        latents = render

        # dummy variables
        elevation = torch.Tensor([0]).to(device=self.device)
        azimuth = torch.Tensor([0]).to(device=self.device)
        camera_distances = torch.Tensor([1]).to(device=self.device)

        # 2. Encode prompt into view dependent text embeddings (equivalent to prompt_embeds)
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        text_embeddings_vd = text_embeddings_vd.unsqueeze(1) #[2,1,77,1024]
        text_embeddings_vd  = torch.cat([text_embeddings_vd]*image.shape[0],dim=1) # [2,B,77,1024]
        text_embeddings_vd = text_embeddings_vd.flatten(0,1) # [2B,77,1024]

        # 3. Encode prompt into view independent text embeddings
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )
        text_embeddings = text_embeddings.unsqueeze(1) #[2,1,77,1024]
        text_embeddings  = torch.cat([text_embeddings]*image.shape[0],dim=1) # [2,B,77,1024]
        text_embeddings = text_embeddings.flatten(0,1) # [2B,77,1024]

        # 4. Add noise to LR image
        noise_level = torch.Tensor([noise_level]).to(dtype=torch.long, device=self.device)
        noise = randn_tensor(image.shape, device=self.device, dtype=text_embeddings.dtype) # draw gaussian noise sample
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        noise_level  = torch.cat([noise_level]*image.shape[0]) # [BB]

        # 5. Check that sizes of lr image and latents match
        assert image.shape[2:] == latents.shape[2:], f"height/width mismatch! {image.shape, latents.shape}"
        num_channels_image = image.shape[1]
        num_channels_latents = self.vae.config.latent_channels

        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )
        
        assert self.cfg.guidance_type in ['sds','rsd']

        if self.cfg.guidance_type == 'rsd':
            latents_noisy, pred_latents_noisy = self.compute_grad_rsd(latents=latents, 
                                        image=image, 
                                        text_embeddings_vd=text_embeddings_vd, 
                                        text_embeddings=text_embeddings, 
                                        noise_level=noise_level)
            return latents_noisy, pred_latents_noisy
        
        if self.cfg.guidance_type == 'sds':
            grad = self.compute_grad_sds(latents=latents, 
                            image=image, 
                            text_embeddings_vd=text_embeddings_vd, 
                            text_embeddings=text_embeddings, 
                            noise_level=noise_level)
            grad = torch.nan_to_num(grad) 
            target = (latents - grad).detach()
            return latents, target    

        raise ValueError(f"Invalid guidance type: {self.cfg.guidance_type}")  
    
    def timestep_annealing(self, stage_step: int):
        # not part of Updateable, needs to be run manually in each training step

        if self.cfg.anneal_strategy == "none":
            self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
            self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
            return

        if (self.cfg.anneal_start_step is not None and stage_step >= self.cfg.anneal_start_step):

            if self.cfg.anneal_strategy == "milestone":
                self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent_annealed)
                return
            
            if self.cfg.anneal_strategy == "sqrt":
                max_step_percent_annealed = self.cfg.max_step_percent - (self.cfg.max_step_percent - self.cfg.min_step_percent) * math.sqrt(
                    (stage_step - self.cfg.anneal_start_step)
                    / (self.cfg.anneal_end_step - self.cfg.anneal_start_step)
                )
                self.max_step = int(self.num_train_timesteps * max_step_percent_annealed)
                self.min_step = self.max_step #non-stochastic, monotonically decreasing t
                return
            
            if self.cfg.anneal_strategy == "linear":
                step_fraction = 1.0-(stage_step - self.cfg.anneal_start_step) / (self.cfg.anneal_end_step - self.cfg.anneal_start_step)
                max_step_percent_annealed = step_fraction*(self.cfg.max_step_percent-self.cfg.min_step_percent)+self.cfg.min_step_percent
                self.max_step = int(self.num_train_timesteps * max_step_percent_annealed)
                self.min_step = self.max_step #non-stochastic, monotonically decreasing t
                return
            
            if self.cfg.anneal_strategy == 'discrete':
                self.cfg.random_timestep = False
                self.step_fraction = (stage_step - self.cfg.anneal_start_step) / (self.cfg.anneal_end_step - self.cfg.anneal_start_step)
                self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
                self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
                return

            raise ValueError(
                f"Unknown anneal strategy {self.cfg.anneal_strategy}, should be one of 'milestone', 'sqrt', 'none'"
            )
        
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def upscale(
        self,
        render: Float[Tensor, "B H W C"],
        image: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput, 
        noise_level=20,
        # **kwargs
        ):  
        
        render = render.permute(0,3,1,2)
        image = image.permute(0,3,1,2)

        # dummy variables
        elevation = torch.Tensor([0]).to(device=self.device)
        azimuth = torch.Tensor([0]).to(device=self.device)
        camera_distances = torch.Tensor([1]).to(device=self.device)

        B = render.shape[0]

        # 1. Encode render into latents
        image = image * 2.0 - 1.0
        render = render * 2.0 - 1.0
        latents = self.encode_images(render)

        # 2. Encode prompt into view dependent text embeddings (equivalent to prompt_embeds)
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        text_embeddings_vd  = torch.cat([text_embeddings_vd]*image.shape[0]) # [BB]

        # 3. Encode prompt into view independent text embeddings
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )
        text_embeddings  = torch.cat([text_embeddings]*image.shape[0]) # [BB]

        # 4. Add noise to LR image
        noise_level = torch.Tensor([noise_level]).to(dtype=torch.long, device=self.device)
        noise = randn_tensor(image.shape, device=self.device, dtype=text_embeddings.dtype) # draw gaussian noise sample
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        noise_level  = torch.cat([noise_level]*image.shape[0]) # [BB]

        # 5. Check that sizes of lr image and latents match
        assert image.shape[2:] == latents.shape[2:], f"height/width mismatch! {image.shape, latents.shape}"
        num_channels_image = image.shape[1]
        num_channels_latents = self.vae.config.latent_channels

        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )
        
        # random timestamp
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [B],
            dtype=torch.long,
            device=self.device,
        )

        # prepare constant timesteps
        last_timestep = t[0].detach().cpu().numpy()
        self.last_timestep = torch.tensor(last_timestep)
        sample_scheduler = self.set_constant_timesteps(self.scheduler_sample, last_timestep, self.cfg.num_inference_steps, device=self.device) # timesteps to apply unet (t+1)
        timesteps = sample_scheduler.timesteps

        # add noise to latents
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, t) #[1,3,256,256] (zt) # to fit
        noise_level = torch.cat([noise_level]*2)


        # 8. Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance

            latent_model_input = torch.cat([latents, image], dim=1)
            latent_model_input = torch.cat([latent_model_input] * 2)
            latent_model_input = sample_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.pipe.unet(
                    latent_model_input.to(self.weights_dtype),
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=noise_level,
            ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond) #cfg

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample #xt-1 #BCHW


        sr_image = self.decode_latents(latents)

        return sr_image
    

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def set_constant_timesteps(
        self,
        scheduler : DPMSolverMultistepScheduler,
        last_timestep: Int,
        num_inference_steps: Int,
        device: str,
        ):

        # define last_timestep in integers
        assert last_timestep < 1000 and last_timestep > 0, f"last_timestep out of accepted range: {last_timestep}"
        timesteps = np.linspace(last_timestep,0,num_inference_steps).round().copy().astype(np.int64)
        timesteps += 1

        sigmas = np.array(((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5)

        scheduler.sigmas = torch.from_numpy(sigmas)

        # when num_inference_steps == num_train_timesteps, we can end up with duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        scheduler.timesteps = torch.from_numpy(timesteps).to(device)

        scheduler.num_inference_steps = len(timesteps)

        scheduler.model_outputs = [None,] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0

        return scheduler