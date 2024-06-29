# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
from PIL import Image, ExifTags
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
        check_orientation: bool = True,
    ):
        image = Image.open(input_file)

        if check_orientation:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = dict(image._getexif().items())

                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (KeyError, AttributeError):
                # EXIF data does not have orientation
                # Do not rotate
                pass

        image.save(os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):

        image_input = workflow["970"]["inputs"]
        image_input["image"] = kwargs["image_url"]

        gender_input = workflow["1738"]["inputs"]
        gender_input["value"] = kwargs["selected_gender"].capitalize()

        background_input = workflow["1718"]["inputs"]
        background_input["text"] = kwargs["background"]

        style_prompt_input = workflow["1719"]["inputs"]
        style_prompt_input["text"] = kwargs["style_prompt"]

        clothing_input_1 = workflow["1716"]["inputs"]
        clothing_input_1["text"] = kwargs["clothing"]

        clothing_input_2 = workflow["1717"]["inputs"]
        clothing_input_2["text"] = kwargs["clothing"]
        pass

    def predict(
        self,
        image: Path = Input(
            description="An input image",
            default=None,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
        image_url: str = Input(
            description="URL of the input image",
            default="",
        ),
        selected_gender: str = Input(
            description="Selected gender",
            default="",
        ),
        background: str = Input(
            description="Background description",
            default="",
        ),
        style_prompt: str = Input(
            description="Style prompt",
            default="",
        ),
        clothing: str = Input(
            description="Clothing description",
            default="",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        if image:
            self.handle_input_file(image)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            image_url=image_url,
            selected_gender=selected_gender,
            background=background,
            style_prompt=style_prompt,
            clothing=clothing,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
