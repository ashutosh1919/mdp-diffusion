import os
import numpy as np
import gradio as gr
from mdp import MDP

mdp_obj = MDP()


def synthetic_image_editing(
    sampler_type: str,
    init_prompt: str,
    edit_prompt: str,
    num_infer_steps: int,
    tmax: int,
    tm: int,
    manip_schedule_type: str,
    enable_classifier_free_guidance: bool,
    guidance_scale: float
):
    img_edit, img_org = mdp_obj.synthetic_image_editing(
        sampler_type=sampler_type,
        init_prompt=init_prompt,
        edit_prompt=edit_prompt,
        num_infer_steps=num_infer_steps,
        tmax=tmax,
        tm=tm,
        manip_schedule_type=manip_schedule_type,
        enable_classifier_free_guidance=enable_classifier_free_guidance,
        guidance_scale=guidance_scale
    )
    return np.asarray(img_org), np.asarray(img_edit)


app_inputs = [
    gr.Dropdown(
        label="Algorithm Type",
        choices=['mdp_epsilon', 'mdp_condition', 'mdp_x'],
        value='mdp_epsilon'
    ),
    gr.Textbox(
        label="Text Prompt to Initialize Image",
        value="Photo of a forest in the spring"
    ),
    gr.Textbox(
        label="Text Prompt to Edit Image",
        value="A forest, winter"
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
    gr.Image(label="Initial Image"),
    gr.Image(label="Edited Image")
]

demo = gr.Interface(
    fn=synthetic_image_editing,
    inputs=app_inputs,
    outputs=app_outputs,
    title="Modifying Diffusion Path: Synthetic Image Editing"
)

demo.queue()
demo.launch(share=True)
