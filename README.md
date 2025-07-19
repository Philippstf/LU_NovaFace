# 🎯 nuva face - AI Lip Enhancement Prediction

MVP eines Machine Learning Systems zur Vorhersage von Lippenunterspritzungs-Ergebnissen.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Airtable Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Airtable credentials:
# AIRTABLE_API_KEY=your_api_key
# AIRTABLE_BASE_ID=your_base_id  
# AIRTABLE_TABLE_NAME=your_table_name
```

### 3. Extract Data from Airtable
```bash
python scripts/extract_data.py
```

### 4. Start Training
```bash
python scripts/train_model.py --epochs 100 --batch-size 8
```

## 📁 Project Structure

```
LU_NovaFace/
├── src/
│   ├── data/              # Data loading & preprocessing
│   │   ├── airtable_extractor.py  # Airtable API integration
│   │   └── dataset.py             # PyTorch dataset & transforms
│   ├── models/            # Neural network architectures  
│   │   ├── generator.py           # Generator models
│   │   ├── discriminator.py       # Discriminator models
│   │   └── gan.py                 # Complete GAN system
│   └── training/          # Training pipeline
│       └── trainer.py             # Training loop & logging
├── data/
│   ├── raw/               # Downloaded images & metadata
│   ├── processed/         # Preprocessed data  
│   └── splits/            # Train/val/test splits
├── scripts/               # Utility scripts
├── configs/               # Configuration files
└── checkpoints/           # Model checkpoints
```

## 🎮 Training Options

```bash
# Basic training
python scripts/train_model.py

# Advanced options
python scripts/train_model.py \
    --epochs 200 \
    --batch-size 16 \
    --lr-g 1e-4 \
    --lr-d 2e-4 \
    --generator unet \
    --discriminator multiscale \
    --image-size 512
```

## 📊 Model Architectures

### Generators
- **ConditionalGenerator**: ResNet-based with condition injection
- **UNetGenerator**: U-Net with skip connections (better detail preservation)

### Discriminators  
- **PatchDiscriminator**: PatchGAN for efficient patch-based classification
- **MultiScaleDiscriminator**: Multi-resolution discrimination

### Loss Functions
- **Adversarial Loss**: GAN objective
- **L1 Loss**: Pixel-wise reconstruction (λ=100)
- **Perceptual Loss**: VGG-based feature matching (λ=10) 
- **Style Loss**: Gram matrix matching (λ=50)

## 🔧 Data Pipeline

1. **Airtable Extraction**: Downloads images and metadata
2. **Preprocessing**: Resize, normalize, augment images
3. **Condition Encoding**: Convert treatment parameters to embeddings
4. **Data Splits**: 80% train, 10% validation, 10% test

## 📈 Monitoring

- **Wandb Integration**: Automatic logging of losses, samples, metrics
- **Checkpoints**: Best model and periodic saves
- **Sample Generation**: Visual progress tracking every 5 epochs

## 🎯 Expected Input/Output

### Input
- **Before Image**: 512x512 RGB image of lip region
- **Treatment Parameters**:
  - Product type (e.g., "hyaluronic_acid")
  - Volume (ml, e.g., 0.7)
  - Region (e.g., "upper_lip")
  - Gender, Age

### Output
- **After Image**: 512x512 RGB predicted result

## 📝 Airtable Schema

Your Airtable should have these columns:
- `Vorher-Foto`: Attachment field (before image)
- `Nachher-Foto`: Attachment field (after image)
- `Produkt`: Single select (product type)
- `Volumen pro Region`: Number (ml volume)
- `Region`: Single select (treatment region)
- `Geschlecht`: Single select (gender)
- `Alter`: Number (age)
- `Notizen`: Long text (notes)

## 🚨 Troubleshooting

### Common Issues

**"No CUDA device found"**
- Install PyTorch with CUDA support
- Check GPU availability with `nvidia-smi`

**"Airtable API Error"**
- Verify API key and base ID in `.env`
- Check table permissions

**"Out of memory"**
- Reduce batch size: `--batch-size 4`
- Reduce image size: `--image-size 256`

### System Requirements
- **GPU**: 8GB+ VRAM recommended
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for data and checkpoints

## 📚 Next Steps

1. **Data Collection**: Increase dataset to 500+ samples
2. **Model Improvements**: Experiment with diffusion models
3. **Evaluation**: Clinical validation with medical experts
4. **Deployment**: REST API with FastAPI
5. **Frontend**: Web interface for easy testing

## 📄 License

MIT License - See LICENSE file for details.

---

**Generated with Claude Code** 🤖