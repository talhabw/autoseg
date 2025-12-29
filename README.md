# AutoSeg

An annotation tool for image segmentation with SAM integration and automatic propagation.

## Features

- Manual bounding box annotation
- SAM-powered automatic segmentation from bounding boxes
- Encoder-based annotation propagation across sequential images
- YOLO-seg format export
- Review workflow for propagated annotations

## Installation

### Prerequisites

- **Python**: 3.12+
- **uv**: recommended
- **CUDA**: 12+
- **Node.js/Bun**: For frontend

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone --recurse-submodules https://github.com/talhabw/autoseg.git
   cd autoseg
   ```

2. **Install Python dependencies**:
   ```bash
   uv sync
   ```
   *Note: This creates a `.venv` with Python 3.12 and installs all required packages.*

3. **Install Frontend**:
   ```bash
   cd frontend
   bun install
   ```

4. **Download Models**:
   Place your model weights in the `models/` directory in the root. The application expects specific filenames:
   - **SAM3**: Downloaded automatically on first use. [(may require hf cli authentication)](https://huggingface.co/facebook/sam3)
   - **DINOv3**: [Download from Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
     - `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` (ViT-H/16)
     - `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (ViT-L/16)
     - `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (ViT-B/16)
   - **Pixio**: [Download from HuggingFace](https://huggingface.co/collections/facebook/pixio) 
     - `pixio_vitb16.pth`
     - `pixio_vitl16.pth`
     - `pixio_vith16.pth`
     - `pixio_vit1b16.pth`

## Usage

**Backend**:
```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 5172
```

**Frontend**:
```bash
cd frontend
bun run build
bun run preview --port 5173
```
Access at: http://localhost:5173

## Testing

Run the test suite (you may need to install optional dependencies from `pyproject.toml`):
```bash
uv run pytest
```

## Acknowledgements

This project builds upon the following excellent third-party projects and model releases:

- **SAM3**: Segment Anything model used for SAM-powered mask generation.
- **DINOv3 / Pixio**: Vision encoder used for feature extraction / propagation.

I am grateful to the authors and maintainers of these projects for releasing their work.

## License

This project depends on third-party components with their own licenses:

- **Pixio**: [Facebook license](https://github.com/facebookresearch/pixio/blob/main/LICENSE).
- **DINOv3**: [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
- **SAM3**: [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

Please review all licenses before use.

Your rights to use, redistribute, or deploy this project “end-to-end” may be constrained by the licenses of the third-party model code and/or weights you choose to use. Please review and comply with the applicable third-party licenses before use, especially for redistribution or commercial deployment.

Where this repository does not directly redistribute third-party weights, users are expected to obtain them separately and agree to the corresponding license terms from the original providers.
