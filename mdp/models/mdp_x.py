from typing import Any, Optional, List
import torch
from mdp.utils import ImageSampler
from tqdm.auto import tqdm
from torch import autocast


class MDPX(ImageSampler):
    def __init__(self,
                 unet: Any,
                 vae: Any,
                 tokenizer: Any,
                 text_encoder: Any,
                 generator: Any = None,
                 scheduler: Any = None,
                 prompt_text: str = "",
                 latent: Optional[torch.Tensor] = None,
                 num_infer_steps: int = 50,
                 strength: int = 1,
                 guidance_scale: float = 7.5,
                 uncond_embs: Optional[List[Any]] = None,
                 enable_classifier_free_guidance: bool = True,
                 eta: float = 0.0,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        This class defines the image sampling using mdp_x
        technique. For more information, please look at the appendix of
        the paper https://arxiv.org/abs/2303.16765

        Args:
          unet: UNet model taken from `StableDiffusionPipeline`.
          vae: Variational AutoEncoder used for reconstructing latents.
          generator: Latent tensor generator used in case when we are passing
            text to generate the image.
          scheduler: Standard `DDIMScheduler` used in diffusion process for
            using different values of params in different timesteps.
          prompt_text: Text denoting the condition for image sampling. Empty
            string can be passed if you are passing `latent` param.
          tokenizer: Text tokenizer taken from `StableDiffusionPipeline`.
          text_encoder: Text encoder taken from `StableDiffusionPipeline`
          latent: Latent noise tensor based on which the image will be sampled.
          num_infer_steps: Number of inference steps.
          strength: Used to define the initial timestep.
          guidance_scale: Parameter affecting the scale of parameter `beta` in
            classifier free guidance equation. (Only valid if
            `enable_classifier_free_guidance` is True).
          uncond_embs: List containing unconditional embeddings corresponding
            to all timesteps extracted during null inversion.
          enable_classifier_free_guidance: Boolean denoting if you want to
            enable classifier free guidance during image sampling.
          eta: Weight of noise to be added during diffusion process.
          device: Device on which torch will run the sampling.
          **kwargs: Other arguments corresponding to specific image sampling.
            It has more arguments like linear_factor, amplify, etc.
        """
        super().__init__(unet,
                         vae,
                         tokenizer,
                         text_encoder,
                         generator,
                         scheduler,
                         prompt_text,
                         latent,
                         num_infer_steps,
                         strength,
                         guidance_scale,
                         uncond_embs,
                         enable_classifier_free_guidance,
                         eta,
                         device,
                         **kwargs)
        self.test_other_param_valid()

    def test_other_param_valid(self):
        # print(self.kwargs)
        if self.prompt_text == "":
            raise ValueError(
                "prompt_text must not be None for MDPEpsilon")
        if 'amplify' not in self.kwargs:
            self.kwargs['amplify'] = 1
            print("WARN: Ineteger param amplify is set to 1.")
        if 'rec_noise_latents' not in self.kwargs:
            raise ValueError("List param rec_noise_latents must be passed.")
        if 'linear_factor_list' not in self.kwargs:
            self.kwargs['linear_factor_list'] = None
        if 'linear_factor' not in self.kwargs:
            self.kwargs['linear_factor'] = 1
            print("WARN: Ineteger param linear_factor is set to 1.")

    @torch.no_grad()
    def inference_step(self,
                       index: int,
                       timestep: Any,
                       text_embeddings: Any,
                       uncond_embeddings: Any,
                       all_embeddings: Any):
        """
        Inference step for Img2ImgSampler.
        """
        with autocast('cuda'):
            latent_model_input = torch.cat([self.latent] * 2) \
                if self.enable_classifier_free_guidance else self.latent
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep)
            if self.uncond_embs:
                all_embeddings = torch.cat(
                    [self.uncond_embs[index], text_embeddings], dim=0)
            else:
                all_embeddings = torch.cat(
                    [uncond_embeddings, text_embeddings], dim=0)
            noise_pred = self.unet(latent_model_input,
                                   timestep,
                                   encoder_hidden_states=all_embeddings).sample

            # perform guidance
            if self.enable_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + \
                    self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        self.latent = self.scheduler.step(noise_pred,
                                          timestep,
                                          self.latent,
                                          eta=self.eta).prev_sample
        if self.kwargs["linear_factor_list"] is None:
            self.latent = self.kwargs["linear_factor"] * self.kwargs[
                "amplify"] * self.kwargs["rec_noise_latents"][index] + \
                (1 - self.kwargs["linear_factor"]) * self.latent
        else:
            self.latent = self.kwargs["linear_factor_list"][index] * \
                self.kwargs["amplify"] * \
                self.kwargs["rec_noise_latents"][index] \
                + (1 - self.kwargs["linear_factor_list"][index]) * \
                self.latent
        return noise_pred, self.latent, all_embeddings
