# --------------------------------------------------------
# This file is a modified version of
# https://github.com/QianWangX/MDP-Diffusion/blob/main/null_text_inversion.py
# The file is changed in order to make it more structured and legible
# --------------------------------------------------------

from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from tqdm.notebook import tqdm
import torch
from diffusers import DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
from torch.optim.adam import Adam
from PIL import Image
from mdp.utils import load_512


NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5


class NullInversion:
    def __init__(self, model: Any):
        """
        This class applies Null Inversion process over the image for MDP.
        It means that it is responsible for producing initial noise from the
        original image x(A) and its default condition c(A).
        """
        scheduler = DDIMScheduler(beta_start=0.00085,
                                  beta_end=0.012,
                                  beta_schedule="scaled_linear",
                                  clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt: Optional[str] = None
        self.context: Optional[torch.Tensor] = None

    @property
    def scheduler(self) -> Any:
        return self.model.scheduler

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        """
        To apply classifier-free guidance, this method creates two different
        embeddings: uncoditional (empty) string and the prompt text string.
        Concatenation of these two is called context.
        """
        uncond_input = self.model.tokenizer(
            [""],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(
            uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def image2latent(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Get batched tensor from image.
        """
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(
                    self.model.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self,
                     latents: torch.Tensor,
                     return_type: str = 'np') -> Any:
        """
        Convert batched tensor to image tensor/numpy array.
        """
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    def get_noise_pred_single(self,
                              latents: Any,
                              t: Any,
                              context: Any) -> Any:
        """
        Get the noised image at timestep `t` with `context` using UNet
        inside the diffusion model.
        """
        noise_pred = self.model.unet(latents,
                                     t,
                                     encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self,
                       latents: Any,
                       t: Any,
                       is_forward: bool = True,
                       context: Optional[Any] = None) -> Any:
        """
        Get the noised latents at timestep `t` with `context` using UNet
        inside the diffusion model.
        """
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input,
                                     t,
                                     encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = (noise_pred_uncond + guidance_scale *
                      (noise_prediction_text - noise_pred_uncond))
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    def prev_step(self,
                  model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]) -> Any:
        """
        Generate the image for the previous step in diffusion process.
        """
        prev_timestep = (timestep - self.scheduler.config.num_train_timesteps
                         // self.scheduler.num_inference_steps)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] \
            if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / \
            alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + \
            pred_sample_direction
        return prev_sample

    def next_step(self,
                  model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int,
                  sample: Union[torch.Tensor, np.ndarray]) -> Any:
        """
        Generate the image for the next step in diffusion process.
        """
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps //
            self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] \
            if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / \
            alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + \
            next_sample_direction
        return next_sample

    @torch.no_grad()
    def ddim_loop(self, latent: torch.Tensor) -> Any:
        """
        Run loop over all DDIM timesteps to collect the latent tensors
        corresponding to images for all those timesteps.
        """
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_inversion(self,
                       image: np.ndarray) -> Tuple[Any, Any]:
        """
        High-level method to apply DDIM loop, to get the reconstructed image
        from diffusion model and all latent vectors for all the timesteps.
        """
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self,
                          latents: Any,
                          num_inner_steps: int,
                          epsilon: float) -> List[Any]:
        """
        Null optimization to apply MDP steps for classifier-free guidance
        based image editing.
        """
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur,
                                                             t,
                                                             cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(
                    latent_cur,
                    t,
                    uncond_embeddings)
                noise_pred = (noise_pred_uncond + GUIDANCE_SCALE *
                              (noise_pred_cond - noise_pred_uncond))
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list

    def invert(self,
               image_path: Union[str, np.ndarray],
               prompt: str,
               offsets: tuple = (0, 0, 0, 0),
               num_inner_steps: int = 10,
               early_stop_epsilon: float = 1e-5,
               verbose: bool = False) -> Tuple[Any, Any, Any]:
        """
        High-level method that invert the image using diffusion steps,
        reconstruct the image and applies classifier-free guidance steps.
        """
        self.init_prompt(prompt)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents,
                                                   num_inner_steps,
                                                   early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
