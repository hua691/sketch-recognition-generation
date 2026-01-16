"""
Multimodal Sketch Generation using Pre-trained Models
Task 3: Training-free sketch generation with CLIP and Stable Diffusion
References: [11] CLIP, [12] Stable Diffusion, [13] CLIPasso, [14] DiffSketcher, [15] Multi-Style
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available. Install with: pip install diffusers")


class BezierCurve:
    """Bezier curve representation for differentiable sketch rendering."""
    
    @staticmethod
    def cubic_bezier(t, p0, p1, p2, p3):
        """Compute point on cubic Bezier curve at parameter t."""
        return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
    
    @staticmethod
    def sample_curve(control_points, num_samples=100):
        """Sample points along Bezier curve."""
        t = torch.linspace(0, 1, num_samples).to(control_points.device)
        t = t.view(-1, 1)
        
        p0, p1, p2, p3 = control_points[0], control_points[1], control_points[2], control_points[3]
        
        points = BezierCurve.cubic_bezier(t, p0, p1, p2, p3)
        return points


class DifferentiableRasterizer(nn.Module):
    """Differentiable rasterizer for stroke-based rendering."""
    
    def __init__(self, canvas_size=224, stroke_width=1.0, device='cuda'):
        super().__init__()
        self.canvas_size = canvas_size
        self.stroke_width = stroke_width
        self.device = device
    
    def forward(self, strokes, stroke_widths=None, stroke_colors=None):
        """
        Render strokes to canvas.
        Args:
            strokes: List of stroke control points [num_strokes, 4, 2]
            stroke_widths: Optional per-stroke widths [num_strokes]
            stroke_colors: Optional per-stroke colors [num_strokes, 3]
        Returns:
            Rendered image [1, 3, H, W]
        """
        canvas = torch.ones(1, 3, self.canvas_size, self.canvas_size).to(self.device)
        
        if stroke_widths is None:
            stroke_widths = torch.ones(len(strokes)).to(self.device) * self.stroke_width
        
        if stroke_colors is None:
            stroke_colors = torch.zeros(len(strokes), 3).to(self.device)
        
        # Create coordinate grid
        y_coords = torch.arange(self.canvas_size).float().to(self.device)
        x_coords = torch.arange(self.canvas_size).float().to(self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        
        for stroke, width, color in zip(strokes, stroke_widths, stroke_colors):
            # Sample points on curve
            points = BezierCurve.sample_curve(stroke, num_samples=50)  # [50, 2]
            
            # Compute distance from each pixel to curve
            # Simplified: use minimum distance to sampled points
            points_scaled = points * self.canvas_size
            
            # Distance to each point
            diff = grid.unsqueeze(2) - points_scaled.unsqueeze(0).unsqueeze(0)  # [H, W, 50, 2]
            dist = torch.norm(diff, dim=-1)  # [H, W, 50]
            min_dist = dist.min(dim=-1)[0]  # [H, W]
            
            # Soft stroke rendering
            alpha = torch.sigmoid((width - min_dist) * 2)  # [H, W]
            alpha = alpha.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Apply color
            color_map = color.view(1, 3, 1, 1).expand(1, 3, self.canvas_size, self.canvas_size)
            canvas = canvas * (1 - alpha) + color_map * alpha
        
        return canvas


class CLIPSketchOptimizer(nn.Module):
    """
    CLIPasso-style sketch generation via CLIP optimization.
    Generates sketches by optimizing stroke parameters to match CLIP embedding.
    """
    
    def __init__(self, num_strokes=16, canvas_size=224, device='cuda'):
        super().__init__()
        
        self.num_strokes = num_strokes
        self.canvas_size = canvas_size
        self.device = device
        
        # Load CLIP model
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            self.clip_model = None
        
        # Rasterizer
        self.rasterizer = DifferentiableRasterizer(canvas_size, device=device)
        
        # Initialize stroke parameters (Bezier control points)
        self.init_strokes()
    
    def init_strokes(self):
        """Initialize stroke control points randomly."""
        # Each stroke: 4 control points, each point: (x, y) in [0, 1]
        control_points = torch.rand(self.num_strokes, 4, 2).to(self.device)
        self.control_points = nn.Parameter(control_points)
        
        # Stroke widths
        widths = torch.ones(self.num_strokes).to(self.device) * 2.0
        self.stroke_widths = nn.Parameter(widths)
    
    def render(self):
        """Render current strokes to image."""
        # Clamp parameters
        points = torch.sigmoid(self.control_points)
        widths = F.softplus(self.stroke_widths)
        
        return self.rasterizer(points, widths)
    
    def get_clip_embedding(self, image):
        """Get CLIP embedding for image."""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not available")
        
        # Normalize for CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        
        image_norm = (image - mean) / std
        
        # Resize if needed
        if image_norm.shape[-1] != 224:
            image_norm = F.interpolate(image_norm, size=224, mode='bilinear')
        
        return self.clip_model.encode_image(image_norm)
    
    def optimize_for_text(self, text_prompt, num_iterations=500, lr=0.1):
        """Optimize strokes to match text prompt."""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not available")
        
        # Get text embedding
        text_tokens = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([self.control_points, self.stroke_widths], lr=lr)
        
        losses = []
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Render
            image = self.render()
            
            # Get image embedding
            image_embedding = self.get_clip_embedding(image)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            # CLIP loss (negative cosine similarity)
            clip_loss = -torch.cosine_similarity(image_embedding, text_embedding).mean()
            
            # Regularization: encourage smooth strokes
            points = torch.sigmoid(self.control_points)
            smoothness_loss = ((points[:, 1:] - points[:, :-1]) ** 2).mean()
            
            # Total loss
            loss = clip_loss + 0.1 * smoothness_loss
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")
        
        return self.render(), losses
    
    def optimize_for_image(self, target_image, num_iterations=500, lr=0.1):
        """Optimize strokes to match target image."""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not available")
        
        # Get target embedding
        with torch.no_grad():
            target_embedding = self.get_clip_embedding(target_image)
            target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        
        optimizer = torch.optim.Adam([self.control_points, self.stroke_widths], lr=lr)
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            image = self.render()
            image_embedding = self.get_clip_embedding(image)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            loss = -torch.cosine_similarity(image_embedding, target_embedding).mean()
            
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")
        
        return self.render()
    
    def get_stroke_sequence(self):
        """Convert optimized strokes to stroke-3 format."""
        points = torch.sigmoid(self.control_points).detach().cpu().numpy()
        
        stroke_sequence = []
        for stroke in points:
            # Sample points along Bezier curve
            t = np.linspace(0, 1, 20)
            curve_points = []
            for ti in t:
                p = (1-ti)**3 * stroke[0] + 3*(1-ti)**2*ti * stroke[1] + \
                    3*(1-ti)*ti**2 * stroke[2] + ti**3 * stroke[3]
                curve_points.append(p)
            
            curve_points = np.array(curve_points)
            
            # Convert to stroke-3 (dx, dy, pen_state)
            for i in range(len(curve_points)):
                if i == 0:
                    dx, dy = curve_points[i]
                else:
                    dx = curve_points[i, 0] - curve_points[i-1, 0]
                    dy = curve_points[i, 1] - curve_points[i-1, 1]
                
                pen_state = 1 if i == len(curve_points) - 1 else 0
                stroke_sequence.append([dx, dy, pen_state])
        
        return np.array(stroke_sequence)


class DiffSketchGenerator(nn.Module):
    """
    DiffSketcher-style text-guided sketch generation.
    Uses Stable Diffusion's latent space for sketch generation.
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device='cuda'):
        super().__init__()
        self.device = device
        
        if DIFFUSERS_AVAILABLE:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(device)
            self.pipe.safety_checker = None
        else:
            self.pipe = None
    
    def generate_sketch_from_text(self, prompt, num_strokes=32, 
                                   num_inference_steps=50, guidance_scale=7.5):
        """Generate sketch from text prompt."""
        if self.pipe is None:
            raise RuntimeError("Stable Diffusion not available")
        
        # Generate image first
        sketch_prompt = f"simple line drawing sketch of {prompt}, black and white, minimal strokes"
        
        with torch.no_grad():
            image = self.pipe(
                sketch_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        return image
    
    def extract_strokes_from_image(self, image, num_strokes=32):
        """Extract stroke sequence from generated image using edge detection."""
        import cv2
        
        # Convert to numpy
        img_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to stroke-3 format
        strokes = []
        for contour in contours[:num_strokes]:
            if len(contour) < 2:
                continue
            
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue
            
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()
            
            if len(approx.shape) == 1:
                continue
            
            # Convert to stroke-3
            for i, point in enumerate(approx):
                if i == 0:
                    dx, dy = point[0] / 255.0, point[1] / 255.0
                else:
                    dx = (point[0] - approx[i-1][0]) / 255.0
                    dy = (point[1] - approx[i-1][1]) / 255.0
                
                pen_state = 1 if i == len(approx) - 1 else 0
                strokes.append([dx, dy, pen_state])
        
        return np.array(strokes) if strokes else np.zeros((1, 3))


class StyleTransferSketch(nn.Module):
    """
    Style-guided sketch generation.
    Generate sketches matching the style of exemplar sketches.
    Reference: [15] Text to Sketch Generation with Multi-Styles
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        if CLIP_AVAILABLE:
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
        else:
            self.clip_model = None
        
        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)
    
    def extract_style(self, style_image):
        """Extract style embedding from exemplar sketch."""
        if self.clip_model is None:
            raise RuntimeError("CLIP not available")
        
        with torch.no_grad():
            style_feat = self.clip_model.encode_image(style_image)
        
        return self.style_encoder(style_feat.float())
    
    def generate_with_style(self, content_prompt, style_embedding, 
                           num_strokes=16, num_iterations=300):
        """Generate sketch with given content and style."""
        optimizer = CLIPSketchOptimizer(num_strokes, device=self.device)
        
        # Get content embedding
        text_tokens = clip.tokenize([content_prompt]).to(self.device)
        with torch.no_grad():
            content_embedding = self.clip_model.encode_text(text_tokens)
        
        # Optimize with both content and style
        opt = torch.optim.Adam([optimizer.control_points, optimizer.stroke_widths], lr=0.1)
        
        for i in range(num_iterations):
            opt.zero_grad()
            
            image = optimizer.render()
            image_embedding = optimizer.get_clip_embedding(image)
            
            # Content loss
            content_loss = -torch.cosine_similarity(
                image_embedding, content_embedding
            ).mean()
            
            # Style loss (simplified)
            image_style = self.style_encoder(image_embedding.float())
            style_loss = F.mse_loss(image_style, style_embedding)
            
            loss = content_loss + 0.5 * style_loss
            loss.backward()
            opt.step()
        
        return optimizer.render(), optimizer.get_stroke_sequence()


def evaluate_sketch_quality(generated_sketch, reference_sketch=None, 
                           clip_model=None, device='cuda'):
    """
    Evaluate generated sketch quality.
    Metrics: CLIP similarity, stroke count, reconstruction error
    """
    metrics = {}
    
    # Stroke count
    if isinstance(generated_sketch, np.ndarray):
        pen_ups = (generated_sketch[:, 2] == 1).sum()
        metrics['num_strokes'] = int(pen_ups)
    
    # CLIP similarity (if reference provided)
    if reference_sketch is not None and clip_model is not None:
        with torch.no_grad():
            gen_feat = clip_model.encode_image(generated_sketch)
            ref_feat = clip_model.encode_image(reference_sketch)
            similarity = torch.cosine_similarity(gen_feat, ref_feat).item()
            metrics['clip_similarity'] = similarity
    
    return metrics


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test CLIPasso-style optimization
    if CLIP_AVAILABLE:
        print("Testing CLIPSketchOptimizer...")
        optimizer = CLIPSketchOptimizer(num_strokes=8, device=device)
        
        # Quick test with fewer iterations
        sketch, losses = optimizer.optimize_for_text("a cat", num_iterations=10)
        print(f"Generated sketch shape: {sketch.shape}")
        print(f"Final loss: {losses[-1]:.4f}")
        
        # Get stroke sequence
        strokes = optimizer.get_stroke_sequence()
        print(f"Stroke sequence shape: {strokes.shape}")
    else:
        print("CLIP not available, skipping test")
