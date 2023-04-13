import os
import numpy as np
import gradio as gr
from mdp import MDP

mdp_obj = MDP()


def real_image_editing(
    image: np.ndarray,
    sampler_type: str,
    edit_prompt: str,
    num_infer_steps: int,
    tmax: int,
    tm: int,
    manip_schedule_type: str,
    enable_classifier_free_guidance: bool,
    guidance_scale: float
):
    img, _ = mdp_obj.real_image_editing(
        img_path=image,
        sampler_type=sampler_type,
        edit_prompt=edit_prompt,
        num_infer_steps=num_infer_steps,
        tmax=tmax,
        tm=tm,
        manip_schedule_type=manip_schedule_type,
        enable_classifier_free_guidance=enable_classifier_free_guidance,
        guidance_scale=guidance_scale
    )
    return np.asarray(img)


app_inputs = [
    gr.Image(label="Input Image"),
    gr.Dropdown(
        label="Algorithm Type",
        choices=['mdp_epsilon', 'mdp_condition', 'mdp_x'],
        value='mdp_epsilon'
    ),
    gr.Textbox(
        label="Text Prompt to Edit Image",
        value="Photo of a zebra"
    ),
    gr.Number(
        label="Number of Inference Steps",
        value=50,
        precision=0
    ),
    gr.Number(
        label="T-max (end timestep)",
        value=47,
        precision=0
    ),
    gr.Number(
        label="T-m (total timesteps = end - start)",
        value=26,
        precision=0
    ),
    gr.Dropdown(
        label="Manipulation Schedule Type",
        choices=['constant', 'linear', 'cosine', 'exp'],
        value='constant'
    ),
    gr.Checkbox(
        label="Enable/Disable Classifier-free Guidance",
        value=True,
    ),
    gr.Number(
        label="Guidance Scale (Valid only when guidance enabled)",
        value=7.5,
        precision=2
    )
]

app_outputs = [
    gr.Image(label="Edited Image")
]

demo = gr.Interface(
    fn=real_image_editing,
    inputs=app_inputs,
    outputs=app_outputs,
    title="Modifying Diffusion Path: Real Image Editing"
)

demo.queue()
demo.launch(share=True)
