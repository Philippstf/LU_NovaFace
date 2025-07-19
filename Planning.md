# 🔬 nuva face - ML Project Planning

## 📋 Übersicht

**Projektname:** nuva face  
**Version:** MVP/Prototyp  
**Ziel:** AI-gestütztes Tool zur Vorhersage von Lippenunterspritzungs-Ergebnissen  
**Phase:** Technische Planung & Prototypentwicklung  

---

## 1. 🎯 Zieldefinition & Anwendungsfall

### Primäres Ziel
Entwicklung eines ML-Systems, das basierend auf einem Vorher-Foto der Lippenregion, Produktart (z.B. Hyaluronsäure) und Volumenmenge (ml) ein realistisches Nachher-Foto generiert.

### Anwendungsfälle
- **Beratungstool für Ärzte:** Visualisierung erwarteter Behandlungsergebnisse
- **Patientenaufklärung:** Realistische Erwartungsbildung vor Eingriffen
- **Schulungsplattform:** Training für medizinisches Personal

### Erfolgsmetriken
- **Perceptual Quality:** LPIPS < 0.15, FID < 50
- **Clinical Accuracy:** Arztbewertung der Realitätsnähe (1-10 Skala) > 7
- **User Acceptance:** 80% positive Bewertung in Usability-Tests

---

## 2. 📊 Datenbeschreibung

### Verfügbare Datensätze
- **Anzahl:** 50 Datensätze (Initial)
- **Format:** Vorher/Nachher-Bildpaare + Metadaten

### Datenstruktur
```
Dataset/
├── images/
│   ├── before/
│   │   ├── patient_001_before.jpg
│   │   └── ...
│   └── after/
│       ├── patient_001_after.jpg
│       └── ...
└── metadata/
    └── treatments.csv
```

### Metadaten-Schema
```csv
patient_id,gender,region,technique,product,volume_ml,notes
001,female,upper_lip,injection,hyaluronic_acid,0.7,"subtle enhancement"
```

### Datenanforderungen für Skalierung
- **Minimum für Training:** 500-1000 Bildpaare
- **Optimal:** 2000+ Bildpaare
- **Diversität:** Verschiedene Hauttypen, Altersgruppen, Volumina

---

## 3. 🏗️ Technischer Projektaufbau

### Repository-Struktur
```
LU_NovaFace/
├── data/
│   ├── raw/           # Original Bilder & Metadaten
│   ├── processed/     # Vorverarbeitete Daten
│   └── splits/        # Train/Val/Test Splits
├── src/
│   ├── data/          # Data Loading & Preprocessing
│   ├── models/        # Modellarchitekturen
│   ├── training/      # Training Scripts
│   ├── evaluation/    # Evaluation & Metrics
│   └── inference/     # Inference Pipeline
├── notebooks/         # Explorative Analyse
├── configs/          # Konfigurationsdateien
├── scripts/          # Utility Scripts
├── tests/            # Unit Tests
└── docs/             # Dokumentation
```

### Technologie-Stack
- **Deep Learning:** PyTorch 2.0+
- **Computer Vision:** OpenCV, PIL, albumentations
- **Data Science:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, wandb
- **Deployment:** FastAPI, Docker
- **Cloud:** Google Cloud Platform (bereits verfügbar)

---

## 4. 🧠 Modellwahl & Architektur

### Ansatz 1: Conditional GANs (Primär)
```python
# Architektur-Konzept
class LipEnhancementGAN:
    def __init__(self):
        self.generator = ConditionalGenerator(
            input_channels=3,      # RGB Bild
            condition_dim=64,      # Eingebettete Behandlungsparameter
            output_channels=3      # RGB Nachher-Bild
        )
        self.discriminator = ConditionalDiscriminator(
            input_channels=6,      # Vorher + Nachher Bild
            condition_dim=64
        )
```

**Vorteile:**
- Bewährte Architektur für Image-to-Image Translation
- Gute Kontrolle über Behandlungsparameter
- Relativ schnelles Training

### Ansatz 2: Diffusion Models (Alternative)
```python
# Architektur-Konzept
class LipEnhancementDiffusion:
    def __init__(self):
        self.unet = ConditionalUNet(
            image_channels=3,
            condition_channels=67,  # Bild + Parameter embedding
            timesteps=1000
        )
```

**Vorteile:**
- State-of-the-art Bildqualität
- Bessere Diversität in Ergebnissen
- Robust gegen Mode Collapse

### Konditionierungs-Strategie
```python
# Parameter-Encoding
def encode_treatment_params(product, volume_ml, region):
    embedding = torch.cat([
        product_embedding[product],      # 32-dim
        volume_normalize(volume_ml),     # 1-dim
        region_one_hot[region]          # 8-dim
    ])
    return embedding  # 41-dim total
```

---

## 5. 🎯 Finetuning-Strategie

### Phase 1: Datenaufbereitung
1. **Bildvorverarbeitung:**
   ```python
   transforms = A.Compose([
       A.Resize(512, 512),
       A.CenterCrop(448, 448),      # Fokus auf Lippenregion
       A.Normalize(mean=[0.5], std=[0.5])
   ])
   ```

2. **Datenaugmentation:**
   ```python
   augmentations = A.Compose([
       A.HorizontalFlip(p=0.5),
       A.RandomBrightnessContrast(p=0.3),
       A.HueSaturationValue(p=0.2),
       A.GaussNoise(p=0.1)
   ])
   ```

### Phase 2: Transfer Learning
- **Basis:** Pre-trained StyleGAN2 oder Stable Diffusion
- **Domain Adaptation:** Graduelle Anpassung auf medizinische Bilder
- **Progressive Training:** Schrittweise Erhöhung der Auflösung

### Phase 3: Specialized Training
```python
# Training-Konfiguration
training_config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 200,
    'loss_weights': {
        'adversarial': 1.0,
        'l1': 100.0,
        'perceptual': 10.0,
        'style': 50.0
    }
}
```

---

## 6. 📈 Evaluation

### Quantitative Metriken
```python
evaluation_metrics = {
    'image_quality': {
        'LPIPS': 'Perceptual similarity',
        'FID': 'Fréchet Inception Distance', 
        'SSIM': 'Structural similarity',
        'PSNR': 'Peak signal-to-noise ratio'
    },
    'clinical_accuracy': {
        'volume_correlation': 'Korrelation ml ↔ sichtbare Veränderung',
        'anatomical_consistency': 'Erhaltung natürlicher Proportionen'
    }
}
```

### Qualitative Bewertung
- **Ärztliche Evaluation:** Panel von 3 Fachärzten bewertet Realitätsnähe
- **Blind-Studie:** Unterscheidung zwischen echten/generierten Nachher-Bildern
- **User Studies:** Patientenfeedback zur Nützlichkeit

### A/B Testing Framework
```python
# Evaluation Pipeline
def evaluate_model(model, test_loader):
    results = {
        'lpips_scores': [],
        'fid_score': calculate_fid(model, test_loader),
        'clinical_ratings': collect_expert_ratings(model)
    }
    return results
```

---

## 7. 🚀 MVP-Konzept

### MVP Umfang (4-6 Wochen)
1. **Basis-Pipeline:** Vorher-Bild → Nachher-Bild (ein Produkttyp)
2. **Simple UI:** Web-Interface für Upload & Parametereingabe
3. **Core Model:** GAN-basiert, 256x256 Auflösung
4. **Basic Evaluation:** LPIPS, FID auf Validierungsset

### MVP Architektur
```
Frontend (React/Vue) 
    ↓ HTTP API
Backend (FastAPI)
    ↓ Model Inference
PyTorch Model (512MB)
    ↓ GPU Processing
Prediction Results
```

### MVP Limitierungen
- Nur Hyaluronsäure-Behandlungen
- Begrenzte Volumenbereiche (0.5-1.5ml)
- Kein Real-time Processing
- Basis-Qualitätskontrolle

---

## 8. 💻 Ressourcen & Tools

### Hardware-Anforderungen
- **Training:** NVIDIA GPU mit ≥16GB VRAM (A100, V100)
- **Inference:** GPU mit ≥8GB VRAM (RTX 3080, T4)
- **Storage:** 100GB+ für Daten und Modelle

### Aktuelle GCP-Setup
```bash
# VM-Spezifikationen
Instance: n1-standard-8 (8 vCPUs, 30GB RAM)
GPU: NVIDIA Tesla T4 (16GB VRAM)
OS: Debian GNU/Linux 11
Storage: 200GB SSD
```

### Development Tools
```bash
# Installierte Software
Node.js: v18+
npm: v9+
Claude Code CLI: Latest
Python: 3.9+
CUDA: 11.8
```

### MLOps Pipeline
```yaml
# CI/CD Workflow
stages:
  - data_validation
  - model_training  
  - model_evaluation
  - deployment_staging
  - production_deploy
```

---

## 9. ⚠️ Offene Risiken

### Technische Risiken
- **Kleine Datenmenge:** 50 Samples reichen nicht für robustes Training
- **Domain Gap:** Unterschiede zwischen Trainings- und realen Anwendungsdaten
- **Overfitting:** Risiko bei begrenztem Datenset
- **Ethische KI:** Bias in Hauttyp-/Altersrepräsentation

### Datenrisiken
```python
# Risiko-Mitigation
data_quality_checks = {
    'image_resolution': 'min 512x512px',
    'lighting_consistency': 'standardized conditions',
    'angle_variation': 'max ±15° deviation',
    'metadata_completeness': '100% required fields'
}
```

### Compliance Risiken
- **DSGVO:** Strenge Anforderungen für Gesundheitsdaten
- **Medizinprodukt-Verordnung:** Mögliche Zulassungspflicht
- **Haftung:** Verantwortung für Behandlungsvorhersagen

### Mitigation-Strategien
1. **Datenaugmentation:** Künstliche Vergrößerung des Datensets
2. **Transfer Learning:** Nutzung vortrainierter Modelle
3. **Uncertainty Quantification:** Konfidenzintervalle für Vorhersagen
4. **Legal Review:** Frühzeitige rechtliche Bewertung

---

## 10. 📅 Next Steps

### Woche 1-2: Setup & Datenaufbereitung
```bash
# Prioritäten
1. Python Environment Setup (PyTorch, CUDA)
2. Datenstruktur erstellen und 50 Samples laden
3. Explorative Datenanalyse (EDA)
4. Baseline-Preprocessing Pipeline
```

### Woche 3-4: Modell-Prototyping  
```bash
# Entwicklung
1. Einfache GAN-Architektur implementieren
2. Training Pipeline setup
3. Erste Trainingsläufe mit kleinem Dataset
4. Evaluation Framework
```

### Woche 5-6: MVP Development
```bash
# Integration
1. Modell-Performance optimieren
2. FastAPI Backend entwickeln
3. Simple Frontend (Streamlit/Gradio)
4. End-to-End Testing
```

### Woche 7-8: Evaluation & Iteration
```bash
# Validierung
1. Qualitative Bewertung durch Experten
2. Quantitative Metriken berechnen
3. Modell-Verbesserungen basierend auf Feedback
4. Dokumentation und Demo-Vorbereitung
```

---

## 🔧 Technische Nächste Schritte

### Sofortige Aktionen (heute)
1. **Python ML Environment:** PyTorch + dependencies installieren
2. **Repository Structure:** Projektstruktur anlegen
3. **Data Pipeline:** Ersten Daten-Loader implementieren
4. **Baseline Model:** Einfache GAN-Architektur setup

### Code-Prioritäten
```python
# 1. Data Pipeline (data/loader.py)
class LipDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, metadata_file):
        # Implementation für Daten-Loading
        pass

# 2. Model Architecture (models/gan.py) 
class LipEnhancementGAN(nn.Module):
    def __init__(self):
        # Generator + Discriminator
        pass

# 3. Training Loop (training/train.py)
def train_model(model, dataloader, epochs):
    # Training-Logik
    pass
```

---

**Erstellt:** 19. Juli 2025  
**Version:** 1.0 - Initial Planning  
**Nächstes Review:** Nach MVP-Completion (6-8 Wochen)