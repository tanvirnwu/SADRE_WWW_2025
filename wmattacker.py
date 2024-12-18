from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import os
from skimage.util import random_noise
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from bm3d import bm3d_rgb
from torchvision.models import vgg16
from torchvision.transforms import ToTensor, Normalize
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor


class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError


class VAEWMAttacker(WMAttacker):
    def __init__(self, model_name, quality=1, metric='mse', device='cpu'):
        if model_name == 'bmshj2018-factorized':
            self.model = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'bmshj2018-hyperprior':
            self.model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018-mean':
            self.model = mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018':
            self.model = mbt2018(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'cheng2020-anchor':
            self.model = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)
        else:
            raise ValueError('model name not supported')
        self.device = device

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((512, 512))
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            out = self.model(img)
            out['x_hat'].clamp_(0, 1)
            rec = transforms.ToPILImage()(out['x_hat'].squeeze().cpu())
            rec.save(out_path)


class GaussianBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = cv2.imread(img_path)
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
            cv2.imwrite(out_path, img)


class GaussianNoiseAttacker(WMAttacker):
    def __init__(self, std):
        self.std = std

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            image = cv2.imread(img_path)
            image = image / 255.0
            # Add Gaussian noise to the image
            noise_sigma = self.std  # Vary this to change the amount of noise
            noisy_image = random_noise(image, mode='gaussian', var=noise_sigma ** 2)
            # Clip the values to [0, 1] range after adding the noise
            noisy_image = np.clip(noisy_image, 0, 1)
            noisy_image = np.array(255 * noisy_image, dtype='uint8')
            cv2.imwrite(out_path, noisy_image)


class BM3DAttacker(WMAttacker):
    def __init__(self):
        pass

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path).convert('RGB')
            y_est = bm3d_rgb(np.array(img) / 255, 0.1)  # use standard deviation as 0.1, 0.05 also works
            plt.imsave(out_path, np.clip(y_est, 0, 1), cmap='gray', vmin=0, vmax=1)


class JPEGAttacker(WMAttacker):
    def __init__(self, quality=80):
        self.quality = quality

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            img.save(out_path, "JPEG", quality=self.quality)


class BrightnessAttacker(WMAttacker):
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness)
            img.save(out_path)


class ContrastAttacker(WMAttacker):
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)
            img.save(out_path)


class RotateAttacker(WMAttacker):
    def __init__(self, degree=30):
        self.degree = degree

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            img = img.rotate(self.degree)
            img.save(out_path)


class ScaleAttacker(WMAttacker):
    def __init__(self, scale=0.5):
        self.scale = scale

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            w, h = img.size
            img = img.resize((int(w * self.scale), int(h * self.scale)))
            img.save(out_path)


class CropAttacker(WMAttacker):
    def __init__(self, crop_size=0.5):
        self.crop_size = crop_size

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            w, h = img.size
            img = img.crop((int(w * self.crop_size), int(h * self.crop_size), w, h))
            img.save(out_path)


class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents,
                                   head_start_step=50 - max(self.noise_step // 20, 1),
                                   guidance_scale=7.5,
                                   generator=generator, )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    img.save(out)

            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                img = Image.open(img_path)
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
                latents = self.pipe.vae.encode(img).latent_dist
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
                noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                if return_latents:
                    ret_latents.append(latents.cpu())

            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            if return_latents:
                return ret_latents

class WPWMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}, saliency_mask=None):
        self.pipe = pipe
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        self.saliency_mask = saliency_mask  # Saliency mask for localized noise injection
        #self.dct_range = (10, 20)  # DCT coefficient range
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

        # Pretrained VGG model for feature extraction
        self.vgg_model = vgg16(pretrained=True).features.eval().to(self.device)
        self.preprocess = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Function to generate noise based on the proposed distributions
    def generate_noise(self, shape, device, sigma, noise_type="Laplace"):
        if noise_type == "Laplace":
            b = sigma / torch.sqrt(torch.tensor(2.0, device=device))
            dist = torch.distributions.Laplace(0, b)
            noise = dist.sample(shape)
        elif noise_type == "Cauchy":
            gamma = sigma
            dist = torch.distributions.Cauchy(0, gamma)
            noise = dist.sample(shape)
        elif noise_type == "Poisson":
            lambda_param = sigma  # Assuming lambda is proportional to sigma
            noise = torch.poisson(torch.full(shape, lambda_param, device=device).float())
            if torch.max(noise) > 0:
                noise = noise / torch.max(noise)  # Normalize to [0, 1]
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        print(f"Generated {noise_type} noise with sigma={sigma}")
        return noise

    def adaptive_noise_level(self, x_w):
        # Adaptive noise level based on watermark strength (tau) and image content
        watermark_strength = self.estimate_watermark_strength(x_w)
        sigma = torch.tensor(self.optimize_sigma(watermark_strength), device=self.device)
        print(f"Adaptive noise level calculated: sigma={sigma}, watermark strength={watermark_strength}")
        return sigma

    # def estimate_watermark_strength(self, x_w):
    #     """
    #     Dynamically estimate the watermark strength based on the energy distribution in the DCT domain.
    #     Args:
    #         x_w (torch.Tensor): Input watermarked image (C, H, W).
    #     Returns:
    #         float: Estimated watermark strength.
    #     """
    #     # Convert to grayscale if the input is a color image
    #     if x_w.shape[0] == 3:
    #         x_w = x_w.mean(dim=0)  # Average over the color channels

    #     # Compute the DCT coefficients
    #     dct_coeffs = torch.fft.fft2(x_w)

    #     # Calculate energy distribution
    #     energy = torch.abs(dct_coeffs)  # Magnitude of the coefficients
    #     total_energy = torch.sum(energy)
    #     cumulative_energy = torch.cumsum(energy.flatten(), dim=0) / total_energy

    #     print (energy)
    #     print (total_energy)
    #     print (cumulative_energy)
    #     # Dynamically determine the range based on cumulative energy
    #     low = (cumulative_energy >= 0.5).nonzero(as_tuple=True)[0].min().item()  # 50% energy threshold
    #     high = (cumulative_energy <= 0.9).nonzero(as_tuple=True)[0].max().item()  # 90% energy threshold

    #     # Compute the watermark strength in the dynamically selected range
    #     watermark_strength = torch.mean(torch.abs(dct_coeffs[low:high, low:high]))
    #     print(f"Dynamic DCT range: ({low}, {high}), Estimated watermark strength: {watermark_strength}")
    #     return watermark_strength.item()


    def estimate_watermark_strength(self, x_w):
        """
        Estimate watermark strength using entropy of the normalized image.
        Args:
            x_w (torch.Tensor): Input watermarked image (C, H, W).
        Returns:
            float: Entropy as a measure of watermark strength.
        """
        # Convert to float32 if necessary
        x_w = x_w.to(torch.float32)

        # Normalize to [0, 1]
        x_w = (x_w - x_w.min()) / (x_w.max() - x_w.min())

        # Compute histogram and entropy
        histogram = torch.histc(x_w, bins=256, min=0, max=1)
        prob = histogram / histogram.sum()
        entropy = -torch.sum(prob * torch.log2(prob + 1e-12))  # Add small epsilon to avoid log(0)
        
        print(f"Estimated watermark strength (entropy): {entropy.item()}")

        return entropy.item()

    def optimize_sigma(self, tau):
        # Prevent very small sigma
        lambda_tradeoff = 0.1
        tau = tau / 10.0  # Normalize tau to [0, 1]
        sigma = max(0.1, min(1.0, tau / (1 + lambda_tradeoff * tau)))
        print(f"Optimized sigma value: {sigma} for tau={tau}")
        return sigma


    def compute_latent_saliency_mask(self, latents):
        """
        Compute a saliency mask using features from a pre-trained VGG network.
        Args:
            img (torch.Tensor): Input image tensor (C, H, W).
        Returns:
            torch.Tensor: Saliency mask of shape (1, 1, H, W).
        """
        img = self.normalize(latents).to(self.device).to(dtype=torch.float32)  # Normalize and add batch dimension
        img.requires_grad_()    
        # Extract VGG features
        features = self.vgg_model(img)  # Shape: (1, C, H, W)
        saliency = torch.sum(features**2, dim=1, keepdim=True)  # Feature magnitude (spatial saliency)
        
        # Normalize saliency mask
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        #Step 6: Interpolate saliency map back to original latent resolution
        original_size = (latents.shape[2], latents.shape[3])  # Original H, W
        saliency = torch.nn.functional.interpolate(saliency, size=original_size, mode="bilinear", align_corners=False)
        print(f"Feature-based saliency mask range: min={saliency.min()}, max={saliency.max()}")
        return saliency.to(latents.dtype)

    def extract_features(self, img):
        """
        Extract features from an image using a pre-trained model.
        Args:
            img (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Extracted feature map.
        """
        img = self.normalize(img).to(self.device)
        with torch.no_grad():
            features = self.vgg_model(img)
        print(f"Extracted features with shape={features.shape}")
        return features

    def compute_localized_wasserstein(self, features_clean, features_reconstructed, mask):
        """
        Compute localized Wasserstein distance within the saliency mask regions.
        Args:
            features_clean (torch.Tensor): Features of the clean image.
            features_reconstructed (torch.Tensor): Features of the reconstructed image.
            mask (torch.Tensor): Saliency mask.
        Returns:
            float: Localized Wasserstein distance.
        """
        mask = torch.nn.functional.interpolate(mask, size=(features_clean.shape[2],features_clean.shape[3]), mode="bilinear", align_corners=False)
        assert features_clean.shape == features_reconstructed.shape, "Feature dimensions mismatch"
        assert features_clean.shape[-2:] == mask.shape[-2:], "Feature and saliency mask dimensions mismatch"


        features_clean_masked = features_clean * mask
        features_reconstructed_masked = features_reconstructed * mask

        clean_flat = features_clean_masked.cpu().detach().numpy().flatten()
        reconstructed_flat = features_reconstructed_masked.cpu().detach().numpy().flatten()

        localized_w_distance = wasserstein_distance(clean_flat, reconstructed_flat)
        print(f"Computed localized Wasserstein distance: {localized_w_distance}")
        return localized_w_distance


    def compute_dssim(self, original, reconstructed):
        """
        Compute DSSIM between original and reconstructed images.
        Args:
            original (torch.Tensor): Original image tensor (C, H, W).
            reconstructed (torch.Tensor): Reconstructed image tensor (C, H, W).
        Returns:
            float: DSSIM value.
        """
        # Ensure images are in the same range [0, 1]
        original = (original - original.min()) / (original.max() - original.min() + 1e-8)
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-8)

        if original.dim() == 4:
            original = original.squeeze(0)
        if reconstructed.dim() == 4:
            reconstructed = reconstructed.squeeze(0)

        # Convert tensors to NumPy arrays
        original_np = original.cpu().permute(1, 2, 0).numpy()
        reconstructed_np = reconstructed.cpu().permute(1, 2, 0).numpy()

        # Compute data range
        data_range = 1.0  # Images are normalized to [0, 1]

        # Compute SSIM
        score, _ = ssim(
            original_np,
            reconstructed_np,
            win_size=7,
            channel_axis=-1,
            data_range=data_range,
            full=True
        )
        score = np.clip(score, 0, 1)  # Ensure SSIM is within [0, 1]
        dssim = 1 - score

        print(f"Computed DSSIM: {dssim}, data_range: {data_range}")
        return dssim


    def compute_reconstruction_error(self, original, reconstructed):
        """
        Compute the reconstruction error between the original and reconstructed images.
        Args:
            original (torch.Tensor): Original clean image.
            reconstructed (torch.Tensor): Reconstructed image.
        Returns:
            float: Mean L2 reconstruction error.
        """
        # Normalize both tensors to [0, 1]
        original = (original - original.min()) / (original.max() - original.min() + 1e-8)
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-8)

        # Compute mean reconstruction error
        error = torch.norm(original - reconstructed, p=2) / original.numel()
        print(f"Mean reconstruction error: {error}")
        return error


    def validate_error_bound(self, original, reconstructed, localized_w_distance, C, alpha, sigma):
        """
        Validate the error bound: || A(tilde{z}) - x || <= C Delta_M^alpha + O(sigma).
        Args:
            original (torch.Tensor): Original clean image.
            reconstructed (torch.Tensor): Reconstructed image.
            localized_w_distance (float): Localized Wasserstein distance Delta_M.
            C (float): Constant in the error bound.
            alpha (float): H\"older continuity parameter.
            sigma (float): Noise level.
        Returns:
            bool: True if error bound is satisfied, False otherwise.
        """
        reconstruction_error = self.compute_reconstruction_error(original, reconstructed)
        noise_term = sigma
        bound = C * (localized_w_distance ** alpha) + noise_term
        print(f"Reconstruction error: {reconstruction_error}, Bound: {bound}")
        return reconstruction_error <= bound

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(prompts_buf,
                                head_start_latents=latents,
                                head_start_step=50 - max(self.noise_step // 20, 1),
                                guidance_scale=7.5,
                                generator=generator)
                images = images[0]
                for img, out, original in zip(images, outs_buf, latents_buf):
                    # Convert image back to tensor
                    reconstructed = torch.tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1) / 255
                    reconstructed = reconstructed.unsqueeze(0).to(self.device).to(dtype=torch.float16)  # Match device and precision


                    print (reconstructed.shape)
                    
                    # Decode the original latents using VAE (ensure proper scaling)
                    original_decoded = self.pipe.vae.decode(original / self.pipe.vae.config.scaling_factor).sample

                    print (original_decoded.shape)

                    # Encode both the clean and reconstructed images to latent space
                    features_clean = self.pipe.vae.encode(original_decoded).latent_dist.mean
                    features_reconstructed = self.pipe.vae.encode(reconstructed).latent_dist.mean

                    # features_clean = self.extract_features(original_decoded)
                    # features_reconstructed = self.extract_features(reconstructed)

                    # Debug feature maps
                    print(f"Features Clean: Mean={features_clean.mean().item()}, Std={features_clean.std().item()}")
                    print(f"Features Reconstructed: Mean={features_reconstructed.mean().item()}, Std={features_reconstructed.std().item()}")

                    # Compute saliency mask
                    saliency = self.saliency_mask if self.saliency_mask is not None else self.compute_latent_saliency_mask(original_decoded)
                    print(f"Saliency Mask: Min={saliency.min()}, Max={saliency.max()}")

                    # Compute localized Wasserstein distance
                    localized_w_distance = self.compute_localized_wasserstein(features_clean, features_reconstructed, saliency)
                    print(f"Computed Localized Wasserstein Distance: {localized_w_distance}")


                    # Compute DSSIM
                    dssim = self.compute_dssim(original_decoded, reconstructed)
                    print(f"DSSIM: {dssim}, Localized Wasserstein: {localized_w_distance}")

                    # Validate error bound
                    if not self.validate_error_bound(original_decoded, reconstructed, localized_w_distance, C=1.0, alpha=0.5, sigma=0.1):
                        print(f"Warning: Error bound violated for {out}.")
                    img.save(out)

            

            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                img = Image.open(img_path)
                img_size = 512  # Default image size
                img = img.resize((img_size, img_size))  # Ensure consistent size
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)

                saliency = self.saliency_mask if self.saliency_mask is not None else self.compute_latent_saliency_mask(img)

                latents = self.pipe.vae.encode(img).latent_dist
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor

                sigma = self.adaptive_noise_level(img)
                noise_type = "Laplace" if sigma < 0.3 else ("Cauchy" if sigma < 0.7 else "Poisson")
                noise = self.generate_noise([1, 4, img.shape[-2] // 8, img.shape[-1] // 8],
                       device=self.device, sigma=sigma, noise_type=noise_type)
                noise_scale = sigma * 0.1  # Reduce noise amplitude dynamically
                noise = noise * noise_scale
                if noise.shape != saliency.shape:
                    saliency = torch.nn.functional.interpolate(saliency, size=noise.shape[-2:], mode='bilinear', align_corners=False)
                
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
                noise = noise / (noise.abs().max() + 1e-8)
                noise = noise * saliency
                print(f"Injected noise with type={noise_type}, sigma={sigma}")
                
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)

                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)

                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                if return_latents:
                    ret_latents.append(latents.cpu())

            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            if return_latents:
                return ret_latents
