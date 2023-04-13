# --------------------------------------------------------
# This file is a modified version of
# https://github.com/QianWangX/MDP-Diffusion/blob/main/utils.py
# The file is changed in order to make it more structured and legible
# --------------------------------------------------------

from typing import Optional, Union, List, Tuple, Any
import torch
from torchvision import transforms as tfms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def load_512(image_path: Any,
             left: int = 0,
             right: int = 0,
             top: int = 0,
             bottom: int = 0) -> np.ndarray:
    """
    Load & apply basic processing on image. If the image is not loaded it, then
    load it first from `image_path` (if it is string).
    """
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, _ = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, _ = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def randn_tensor(
    shape: tuple,
    generator: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
) -> torch.Tensor:
    """
    This is a helper function that allows to create random tensors on
    the desired `device` with the desired `dtype`. When passing a list of
    generators one can seed each batched size individually. If CPU generators
    are passed the tensor will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        partial_latents = [
            torch.randn(shape,
                        generator=generator[i],
                        device=rand_device,
                        dtype=dtype,
                        layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(partial_latents, dim=0).to(device)
    else:
        latents = torch.randn(shape,
                              generator=generator,
                              device=rand_device,
                              dtype=dtype,
                              layout=layout).to(device)

    return latents


def pil_to_latent(vae: Any,
                  input_im: Image.Image,
                  device: torch.device,
                  generator: Optional[torch.Generator] = None) -> Any:
    """
    Converting PIL Image to batched object. For example, single image is passed
    and it is converted to tensor of shape say [1, 4, 64, 64].
    """
    latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(device)*2-1)
    latent = 0.18215 * latent.latent_dist.sample(generator)
    return latent


def latents_to_pil(vae: Any,
                   latents: torch.Tensor) -> List[Image.Image]:
    """
    Converting batched tensor to list of PIL images.
    """
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def latents_to_img(vae: Any,
                   latents: torch.Tensor,
                   normalize: str = '0-1',
                   size: int = 512) -> Any:
    """
    Convert encoded batched latent tensor to batched image tensor
    (Tensor containing resized images ready to display)
    """
    latents = (1 / 0.18215) * latents
    image = vae.decode(latents).sample
    if normalize == '0-1':
        image = (image / 2 + 0.5)
    image = torch.nn.functional.interpolate(image,
                                            size=(size, size),
                                            mode='bilinear',
                                            align_corners=False)
    return image


def img_to_latents(vae: Any,
                   input_im: torch.Tensor,
                   generator: Optional[torch.Generator] = None) -> Any:
    """
    Convert batched image tensor to encoded batched latent tensor.
    """
    latent = vae.encode(input_im)
    latent = 0.18215 * latent.latent_dist.sample(generator)
    return latent


def get_timesteps(scheduler: Any,
                  num_inference_steps: int,
                  strength: float) -> Tuple[Any, int]:
    """
    Get the time steps for MDP.
    """
    init_timestep = min(int(num_inference_steps * strength),
                        num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]
    return timesteps, num_inference_steps - t_start


def prepare_latents(vae: Any,
                    scheduler: Any,
                    batch_size: int,
                    num_channels_latents: int,
                    height: int,
                    width: int,
                    device: torch.device,
                    generator: Optional[torch.Generator] = None) -> Any:
    """
    Prepare latent batched tensor to use in diffusion steps for MDP.
    """
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    shape = (batch_size,
             num_channels_latents,
             height // vae_scale_factor, width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=device)
    latents = latents * scheduler.init_noise_sigma
    return latents


def encode_text(prompt: Union[str, List[str]],
                tokenizer: Any,
                text_encoder: Any,
                device: torch.device) -> Any:
    """
    Generate embeddings of the text prompts to provide for editing the image.
    """
    text_input = tokenizer(prompt,
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    text_embeddings = text_embeddings.clone()
    return text_embeddings


def visualize_latents(vae: Any,
                      latents: torch.Tensor,
                      title: str = 'Image'):
    """
    Plot the latent batched tensor on the plot using matplotlib.
    """
    model_output_img = latents_to_pil(vae, latents)
    plt.imshow(model_output_img[0])
    plt.title(title)
    plt.show()


def visualize_images(images: List[Image.Image],
                     titles: Optional[List[str]] = None,
                     cols: int = 5,
                     figsize: tuple = (3, 3)):
    """
    Visualize list of PIL Image objects.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if n_images == 1:
        images[0].show()
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=figsize)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(int(np.ceil(n_images/float(cols))),
                            int(cols), n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def create_linear_factors_list(num_infer_steps: int,
                               tmax: int,
                               tmin: int,
                               manip_type: str = 'constant',
                               constant: int = 1):
    manipulation_range = tmax - tmin
    linear_factors = np.zeros((num_infer_steps, ))
    if manip_type == 'linear':
        linear_factors[num_infer_steps - tmax: num_infer_steps - tmin] = \
            np.linspace(1, 0, manipulation_range)
    elif manip_type == 'cosine':
        linear_factors[num_infer_steps - tmax: num_infer_steps - tmin] = \
            np.cos(np.linspace(0, np.pi / 2, manipulation_range))
    elif manip_type == 'exp':
        linear_factors[num_infer_steps - tmax: num_infer_steps - tmin] = \
            np.exp(np.linspace(0, -5, manipulation_range))
    elif manip_type == 'constant':
        linear_factors[num_infer_steps - tmax: num_infer_steps - tmin] = \
            np.ones(manipulation_range) * constant
    return linear_factors
