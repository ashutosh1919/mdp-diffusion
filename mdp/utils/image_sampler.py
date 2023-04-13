from abc import ABC, abstractmethod
import gc
from typing import Any, Optional, List
import torch
from tqdm.auto import tqdm
from torch import autocast
import numpy as np
from mdp.utils.utils import (
    get_timesteps,
    prepare_latents,
    encode_text,
    latents_to_pil
)


class ImageSampler(ABC):
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
        This abstract class defines the base of the diffusion image
        reconstruction scheme. We will leave one method abstract so that it
        can be implemented for different sampling. It will reduce the
        redundancy.

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
            It will be used in instantiated classes.
        """
        self.unet = unet
        self.vae = vae
        self.generator = generator
        self.scheduler = scheduler
        self.prompt_text = prompt_text
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.latent = latent
        self.num_infer_steps = num_infer_steps
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.uncond_embs = uncond_embs
        self.enable_classifier_free_guidance = enable_classifier_free_guidance
        self.eta = eta
        self.kwargs = kwargs

        if not device:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self.device = device

        # Test parameters for validity
        self.test_basic_param_valid()

    def test_basic_param_valid(self):
        if self.prompt_text == "" and self.latent is None:
            raise ValueError("Either you should pass valid text prompt",
                             "Or you should pass valid latent")
        if self.vae is None:
            raise ValueError("vae cannot be None")
        if self.unet is None:
            raise ValueError("unet cannot be None")
        if self.tokenizer is None:
            raise ValueError("scheduler cannot be None")
        if self.text_encoder is None:
            raise ValueError("scheduler cannot be None")

    @abstractmethod
    def inference_step(self,
                       index: int,
                       timestep: Any,
                       text_embeddings: Any,
                       uncond_embeddings: Any,
                       all_embeddings: Any):
        """
        Implement this in a seperate child class for changing the inference
        step of basic sampling.
        """
        pass

    @torch.no_grad()
    def __call__(self, return_metadata: bool = False):
        """
        Sample image based on the inputs and return the final image.
        It will also return lists corresponding to noise, latents and
        embeddings containing all timesteps of inference of `return_metadata`
        is True.
        """
        # Set timesteps to scheduler
        self.scheduler.set_timesteps(self.num_infer_steps)
        timesteps, num_inference_steps = get_timesteps(
            self.scheduler,
            self.num_infer_steps,
            self.strength)

        # Initialize latent if it is None
        if self.latent is None:
            self.latent = prepare_latents(self.vae,
                                          self.scheduler,
                                          1,
                                          self.unet.in_channels,
                                          height=512,
                                          width=512,
                                          device=self.device,
                                          generator=self.generator)

        # Prepare text embeddings
        text_embeddings = encode_text(self.prompt_text,
                                      self.tokenizer,
                                      self.text_encoder,
                                      self.device)
        uncond_embeddings = encode_text([""],
                                        self.tokenizer,
                                        self.text_encoder,
                                        self.device)
        all_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        if not self.enable_classifier_free_guidance:
            all_embeddings = all_embeddings[1].unsqueeze(0)

        if self.uncond_embs:
            text_embeddings = encode_text(self.prompt_text,
                                          self.tokenizer,
                                          self.text_encoder,
                                          self.device)

        self.unet.eval()
        timesteps = self.scheduler.timesteps
        noise_preds = []
        noise_latents = []
        noise_embs = []

        # Running inference timesteps
        for i, t in enumerate(tqdm(timesteps)):
            pred, latent, embs = self.inference_step(i,
                                                     t,
                                                     text_embeddings,
                                                     uncond_embeddings,
                                                     all_embeddings)
            noise_pred = pred
            self.latent = latent
            all_embeddings = embs
            noise_preds.append(pred)
            noise_latents.append(latent)
            noise_embs.append(embs)

        if return_metadata:
            return (
                latents_to_pil(self.vae, self.latent)[0],
                noise_preds,
                noise_latents,
                noise_embs
            )
        return latents_to_pil(self.vae, self.latent)[0]
