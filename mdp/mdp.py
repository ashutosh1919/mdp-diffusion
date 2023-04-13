from typing import Optional, Dict, Any, Union
import torch
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
import numpy as np
from mdp.utils import (
    load_512,
    create_linear_factors_list,
    Img2ImgSampler,
    Text2ImgSampler
)
from mdp.models import (
    MDPEpsilon,
    MDPCondition,
    MDPX,
    MDPBeta,
    NullInversion
)

sampler_model_mapping: Dict[str, Any] = {
    'mdp_epsilon': MDPEpsilon,
    'mdp_condition': MDPCondition,
    'mdp_x': MDPX,
    'mdp_beta': MDPBeta
}


class MDP:
    def __init__(self,
                 scheduler_beta_start: float = 0.00085,
                 scheduler_beta_end: float = 0.012,
                 scheduler_beta_schedule: str = "scaled_linear",
                 scheduler_clip_sample: bool = False,
                 scheduler_set_alpha_to_one: bool = False,
                 diffusion_model_type: str = "CompVis/stable-diffusion-v1-4",
                 is_synthetic_editing: bool = False,
                 device: Optional[str] = None):
        """
        This class combines all the sampling methods and creates usable
        user interface to use it for real as well as synthetic image editing.

        Args:
          scheduler_beta_start: float param for `DDIMScheduler`.
          scheduler_beta_end: float param for `DDIMScheduler`.
          scheduler_beta_schedule: str param for `DDIMScheduler`.
          scheduler_clip_sample: bool param for `DDIMScheduler`.
          scheduler_set_alpha_to_one: bool param for `DDIMScheduler`.
          diffusion_model_type: str param to create `StableDiffusionPipeline`.
          is_synthetic_editing: bool representing if the input image is
            passed or generated using diffusion model with text prompt.
          device: str representing the device on which to run the model.
        """
        self.scheduler = DDIMScheduler(
            beta_start=scheduler_beta_start,
            beta_end=scheduler_beta_end,
            beta_schedule=scheduler_beta_schedule,
            clip_sample=scheduler_clip_sample,
            set_alpha_to_one=scheduler_set_alpha_to_one
        )

        if device is None:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = StableDiffusionPipeline.from_pretrained(
            diffusion_model_type,
            scheduler=self.scheduler
        ).to(self.device)
        self.vae = self.model.vae
        self.tokenizer = self.model.tokenizer
        self.text_encoder = self.model.text_encoder
        self.unet = self.model.unet

        self.vae = self.vae.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.unet = self.unet.to(self.device)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()

    def real_image_editing(self,
                           img_path: Union[str, np.ndarray],
                           edit_prompt: str,
                           init_prompt: str = "",
                           sampler_type: str = "mdp_epsilon",
                           num_infer_steps: int = 50,
                           tmax: int = 47,
                           tm: int = 26,
                           manip_schedule_type: str = 'constant',
                           strength: int = 1,
                           guidance_scale: float = 7.5,
                           enable_classifier_free_guidance: bool = True,
                           amplify: int = 1,
                           linear_factor: int = 1,
                           eta: float = 0.0,
                           verbose: bool = False):
        """
        Editing the input image based on the inputted prompt
        text. Note that we will use the passed parameters both in
        case of initial latent extraction as well as to sample
        edited image.

        Args:
          img_path: str representing input image path.
          edit_prompt: str representing description of editing prompt.
          init_prompt: (Optional) str describing the input image.
          sampler_type: str representing sampling type among these model types:
            [`mdp_epsilon`, `mdp_condition`, `mdp_x`]
          num_infer_steps: Number of inference steps.
          tmax: int representing T_max or end timestep to modify
            diffusion path.
          tm: int representing T_m or the lenghth of timesteps to modify
            for diffusion path. start timestep is (tm - tmax)
          manip_schedule_type: str representing manipulation schedule to
            select modifying timesteps.
          strength: Used to define the initial timestep.
          guidance_scale: Parameter affecting the scale of parameter `beta` in
            classifier free guidance equation. (Only valid if
            `enable_classifier_free_guidance` is True).
          enable_classifier_free_guidance: Boolean denoting if you want to
            enable classifier free guidance during image sampling.
          eta: Weight of noise to be added during diffusion process.
          amplify: int representing amplifying constant.
          linear_factor: int representing the linear factor for modifying
            condition, noise, or embedding.
          verbose: bool representing whether to print flow of execution.
        """
        if manip_schedule_type not in ['constant', 'linear', 'cosine', 'exp']:
            raise ValueError(
                'manip_schedule_type is not valid. Please pass from {}'.format(
                    ['constant', 'linear', 'cosine', 'exp'])
            )
        if sampler_type not in sampler_model_mapping:
            raise ValueError("{} is not a valid sampler type.".format(
                sampler_type))
        sampler_cls = sampler_model_mapping[sampler_type]

        null_inversion = NullInversion(self.model)
        ((image_gt, image_enc),
         ddim_latents,
         uncond_embeddings) = null_inversion.invert(img_path,
                                                    init_prompt,
                                                    offsets=(0, 0, 0, 0),
                                                    verbose=verbose)

        im2imobj = Img2ImgSampler(
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            latent=ddim_latents,
            uncond_embs=uncond_embeddings,
            num_infer_steps=num_infer_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            enable_classifier_free_guidance=enable_classifier_free_guidance,
            eta=0.0,
            device=self.device
        )
        image, noise_preds, noise_latents, noise_embs = im2imobj(
            return_metadata=True)

        linear_factor_list = create_linear_factors_list(
            num_infer_steps=num_infer_steps,
            tmax=tmax,
            tmin=tmax - tm,
            manip_type=manip_schedule_type
        )

        kwargs = {
            'rec_noise_preds': noise_preds,
            'rec_noise_embs': noise_embs,
            'rec_noise_latents': noise_latents,
            'linear_factor_list': linear_factor_list,
            'amplify': amplify,
            'linear_factor': linear_factor
        }
        mdp_obj = sampler_cls(
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            latent=ddim_latents,
            uncond_embs=uncond_embeddings,
            prompt_text=edit_prompt,
            num_infer_steps=num_infer_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            enable_classifier_free_guidance=enable_classifier_free_guidance,
            device=self.device,
            **kwargs
        )
        edited_image = mdp_obj()
        return edited_image, image

    def synthetic_image_editing(self,
                                init_prompt: str,
                                edit_prompt: str,
                                sampler_type: str = "mdp_epsilon",
                                random_gen_seed: int = 0,
                                num_infer_steps: int = 50,
                                tmax: int = 47,
                                tm: int = 26,
                                manip_schedule_type: str = 'constant',
                                strength: int = 1,
                                guidance_scale: float = 7.5,
                                enable_classifier_free_guidance: bool = True,
                                amplify: int = 1,
                                linear_factor: int = 1,
                                eta: float = 0.0,
                                verbose: bool = False):
        """
        Create the input image based on the initial prompt and then edit
        the created image based on edit prompt. Note that we will use
        the passed parameters both in case of initial latent extraction
        as well as to sample edited image.

        Args:
          init_prompt: str representing description of input image.
          edit_prompt: str representing description of editing prompt.
          sampler_type: str representing sampling type among these model types:
            [`mdp_epsilon`, `mdp_condition`, `mdp_x`]
          random_gen_seed: int representing random seed used for generator.
          num_infer_steps: Number of inference steps.
          tmax: int representing T_max or end timestep to modify
            diffusion path.
          tm: int representing T_m or the lenghth of timesteps to modify
            for diffusion path. start timestep is (tm - tmax)
          manip_schedule_type: str representing manipulation schedule to
            select modifying timesteps.
          strength: Used to define the initial timestep.
          guidance_scale: Parameter affecting the scale of parameter `beta` in
            classifier free guidance equation. (Only valid if
            `enable_classifier_free_guidance` is True).
          enable_classifier_free_guidance: Boolean denoting if you want to
            enable classifier free guidance during image sampling.
          eta: Weight of noise to be added during diffusion process.
          amplify: int representing amplifying constant.
          linear_factor: int representing the linear factor for modifying
            condition, noise, or embedding.
          verbose: bool representing whether to print flow of execution.
        """
        if manip_schedule_type not in ['constant', 'linear', 'cosine', 'exp']:
            raise ValueError(
                'manip_schedule_type is not valid. Please pass from {}'.format(
                    ['constant', 'linear', 'cosine', 'exp'])
            )
        if sampler_type not in sampler_model_mapping:
            raise ValueError("{} is not a valid sampler type.".format(
                sampler_type))
        sampler_cls = sampler_model_mapping[sampler_type]

        generator1 = torch.Generator(device=self.device).manual_seed(
            random_gen_seed)
        tx2imobj = Text2ImgSampler(
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            generator=generator1,
            prompt_text=init_prompt,
            num_infer_steps=num_infer_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            enable_classifier_free_guidance=enable_classifier_free_guidance,
            eta=eta,
            device=self.device
        )
        image, noise_preds, noise_latents, noise_embs = tx2imobj(
            return_metadata=True)

        linear_factor_list = create_linear_factors_list(
            num_infer_steps=num_infer_steps,
            tmax=tmax,
            tmin=tmax - tm,
            manip_type=manip_schedule_type
        )

        generator2 = torch.Generator(device=self.device).manual_seed(
            random_gen_seed)
        kwargs = {
            'rec_noise_preds': noise_preds,
            'rec_noise_embs': noise_embs,
            'rec_noise_latents': noise_latents,
            'linear_factor_list': linear_factor_list,
            'amplify': amplify,
            'linear_factor': linear_factor
        }
        mdp_obj = sampler_cls(
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            generator=generator2,
            prompt_text=edit_prompt,
            num_infer_steps=num_infer_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            enable_classifier_free_guidance=enable_classifier_free_guidance,
            device=self.device,
            **kwargs
        )
        edited_image = mdp_obj()
        return edited_image, image
