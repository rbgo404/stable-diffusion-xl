import torch
from diffusers import DiffusionPipeline
import base64
from io import BytesIO


class InferlessPythonModel:
    """
    Class for loading and using Stable Diffusion XL models for image generation.
    """

    def initialize(self):
        """
        Loads the base and refiner models, optimizes for performance, and prepares for inference.
        """

        # Load models with optimized settings for memory and speed
        load_params = dict(
            torch_dtype=torch.float16,  # Use half-precision for faster inference and less memory
            use_safetensors=True,      # Enable SafeTensors for potential performance benefits
            variant="fp16",            # Specify half-precision variant of the model
            device_map="auto",         # Automatically distribute model across available devices
        )

        # Load the base model (responsible for initial image generation)
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_params
        )

        # Load the refiner model (responsible for refining the generated image)
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,  # Share text encoder for efficiency
            vae=self.base.vae,                       # Share VAE for efficiency
            **load_params,
        )

        # Uncomment these lines if needed for compatibility with your hardware:
        # self.base.enable_model_cpu_offload()  # Offload parts of the model to CPU if needed
        # self.refiner.enable_model_cpu_offload()

        # Compile parts of the model for further optimization (might increase initial inference time)
        self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    def infer(self, inputs):
        """
        Generates an image based on a given prompt using the loaded models.
        """

        prompt = inputs["prompt"]  # Extract the prompt from the input
        negative_prompt = "low resolution"  # Avoid generating low-resolution images
        n_steps = 24  # Number of inference steps (higher values may produce more detailed images)
        high_noise_frac = 0.8  # Fraction of noise at which to switch between base and refiner models

        # Generate a latent representation of the image using the base model
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        # Refine the latent representation using the refiner model
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        # Convert the generated image to base64 encoded string for output
        buff = BytesIO()
        image.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        return {"generated_image_base64": img_str}

    def finalize(self, args):
        """
        Cleans up model resources when inference is finished.
        """

        self.base = None
        self.refiner = None
