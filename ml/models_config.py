"""
Configuration for vision encoder models (DINOv3 and Pixio).
"""

DINOV3_MODELS = {
    "vitb16": {
        "hub_name": "dinov3_vitb16",
        "weights": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "embed_dim": 768,
        "description": "DINOv3 ViT-B/16 (Fast, ~100MB)",
        "download_url": "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/",
    },
    "vitl16": {
        "hub_name": "dinov3_vitl16",
        "weights": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "embed_dim": 1024,
        "description": "DINOv3 ViT-L/16 (Balanced, ~300MB)",
        "download_url": "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/",
    },
    "vith16": {
        "hub_name": "dinov3_vith16plus",
        "weights": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "embed_dim": 1280,
        "description": "DINOv3 ViT-H/16 (Best, ~600MB)",
        "download_url": "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/",
    },
}

PIXIO_MODELS = {
    "pixio_vitb16": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "weights": "pixio_vitb16.pth",
        "description": "Pixio ViT-B/16 (Fast, ~86MB)",
        "download_url": "https://huggingface.co/facebook/pixio-vitb16/resolve/main/pixio_vitb16.pth",
    },
    "pixio_vitl16": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "weights": "pixio_vitl16.pth",
        "description": "Pixio ViT-L/16 (Balanced, ~303MB)",
        "download_url": "https://huggingface.co/facebook/pixio-vitl16/resolve/main/pixio_vitl16.pth",
    },
    "pixio_vith16": {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "weights": "pixio_vith16.pth",
        "description": "Pixio ViT-H/16 (Strong, ~631MB)",
        "download_url": "https://huggingface.co/facebook/pixio-vith16/resolve/main/pixio_vith16.pth",
    },
    "pixio_vit1b16": {
        "embed_dim": 1536,
        "depth": 48,
        "num_heads": 24,
        "weights": "pixio_vit1b16.pth",
        "description": "Pixio ViT-1B/16 (Huge, ~1.4GB)",
        "download_url": "https://huggingface.co/facebook/pixio-vit1b16/resolve/main/pixio_vit1b16.pth",
    },
}

ALL_MODELS = {**DINOV3_MODELS, **PIXIO_MODELS}
