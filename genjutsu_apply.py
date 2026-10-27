"""Task LoRAs and native aligned guides for the Genjutsu presets."""
import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
from comfy_extras.nodes_model_advanced import ModelAttentionBackend
from nodes import LoraLoaderModelOnly

from .genjutsu_nodes import BACKGROUNDS, METHODS

CHARACTER_LORA = "minimax_h3_character_swap_v1.1_ref2va_fro.safetensors"
MOTION_LORA = "minimax_h3_motion_transfer_v1.0_ref2va.safetensors"
TRAINING_ADAPTER = "minimax_h3_ref2va_training_adapter_v2.safetensors"


class SECoursesH3GenjutsuApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        loras = folder_paths.get_filename_list("loras")
        return io.Schema(
            node_id="SECoursesH3GenjutsuApply", display_name="MiniMax H3 Genjutsu Apply",
            category="SECourses/references",
            inputs=[
                io.Model.Input("model"), io.Conditioning.Input("positive"),
                io.Latent.Input("latent"), io.Vae.Input("vae"),
                io.String.Input("method", default=METHODS[0], force_input=True),
                io.String.Input("background", default=BACKGROUNDS[0], force_input=True),
                io.Combo.Input("character_lora", options=loras, default=CHARACTER_LORA),
                io.Combo.Input("motion_lora", options=loras, default=MOTION_LORA),
                io.Combo.Input("training_adapter", options=loras, default=TRAINING_ADAPTER),
                io.Image.Input("guide", optional=True),
            ],
            outputs=[io.Model.Output(), io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, model, positive, latent, vae, method=METHODS[0], background=BACKGROUNDS[0],
                character_lora=CHARACTER_LORA, motion_lora=MOTION_LORA,
                training_adapter=TRAINING_ADAPTER, guide=None):
        if method not in METHODS or background not in BACKGROUNDS:
            raise ValueError("Use the method and background outputs of the Genjutsu node.")
        if method == METHODS[0]:
            return io.NodeOutput(model, positive)
        files = [character_lora] if background == BACKGROUNDS[0] else [training_adapter, motion_lora]
        loader = LoraLoaderModelOnly()
        for filename in files:
            model = loader.load_lora_model_only(model, filename, 1.0)[0]
        if method == METHODS[2]:
            if guide is None:
                raise ValueError("The aligned method needs the Genjutsu node's aligned_guide output connected to guide.")
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae, image=guide).args[0]
            model = ModelAttentionBackend.execute(model, "comfy kitchen attention").args[0]
        print(f"[SECoursesH3Genjutsu] {method}: " + ", ".join(files), flush=True)
        return io.NodeOutput(model, positive)


NODE_CLASS_MAPPINGS = {"SECoursesH3GenjutsuApply": SECoursesH3GenjutsuApply}
