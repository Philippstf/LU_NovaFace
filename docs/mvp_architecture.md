# Nuva Face MVP - Technische Architektur

## 🎯 Ziel
Ein System, das aus einem Eingabebild der Lippen ein realistisches Bild nach einer Hyaluronsäure-Behandlung generiert.

## 🏗️ Pipeline-Architektur

### Stage 1: Preprocessing
```python
Input Image → Face Detection → Lip Segmentation → Landmark Detection → Crop & Normalize
```

**Komponenten:**
- **MediaPipe Face Mesh** für Gesichtserkennung
- **Custom Lip Segmentation** mit U-Net
- **Landmark Detection** für präzise Lippenkontur
- **Normalisierung** auf 256x256px Lippenregion

### Stage 2: Feature Extraction
```python
Lip Image → Encoder → Feature Vector [volume, product, shape_features]
```

**Features:**
- Geometrische Features (Breite, Höhe, Volumen)
- Produkt-Encoding (Restylane Kysse=0, Redensity II=1, etc.)
- Volumen-Parameter (0.7ml, 1ml)

### Stage 3: Conditional Generation
```python
[Lip Features + Treatment Parameters] → Generator → Enhanced Lip Image
```

**Model-Optionen:**
1. **Conditional GAN** (MVP-Empfehlung)
2. **Pix2Pix** für Image-to-Image Translation
3. **ControlNet** für feinere Kontrolle

### Stage 4: Postprocessing
```python
Generated Lip → Blend with Original Face → Color/Lighting Adjustment → Final Image
```

## 📊 Datenstruktur für Training

### Input Format
```python
{
    "before_image": "person_001/before/before_01.jpg",
    "after_image": "person_001/after/after_01.jpg", 
    "volume_ml": 1.0,
    "product": "restylane_kysse",
    "lip_landmarks": [[x1,y1], [x2,y2], ...],
    "treatment_notes": "1. Unterspritzung"
}
```

### Data Augmentation
- Rotation (-10° bis +10°)
- Color Jittering
- Lighting Variation
- Horizontal Flip (mit Landmark-Anpassung)

## 🚀 MVP Implementation Plan

### Phase 1: Data Pipeline (Woche 1)
```python
# 1. Lip Detection & Segmentation
scripts/preprocess_lips.py
# 2. Feature Extraction  
scripts/extract_lip_features.py
# 3. Dataset Creation
scripts/create_training_dataset.py
```

### Phase 2: Model Training (Woche 2-3)
```python
# 1. Conditional GAN Training
src/models/conditional_gan.py
# 2. Training Loop
scripts/train_lip_augmentation.py
# 3. Evaluation Metrics
scripts/evaluate_model.py
```

### Phase 3: Inference Pipeline (Woche 4)
```python
# 1. End-to-End Prediction
src/inference/lip_augmentation_predictor.py
# 2. Web API
app/api/predict.py
# 3. Frontend Interface
app/frontend/upload_interface.html
```

## 🔧 Technische Spezifikationen

### Model Architecture
```python
# Generator: U-Net with Conditional Input
class LipAugmentationGenerator(nn.Module):
    def __init__(self):
        # Encoder: 256x256 → 8x8
        # Conditional Injection: [volume, product] → embedding
        # Decoder: 8x8 → 256x256
        
# Discriminator: PatchGAN
class LipDiscriminator(nn.Module):
    # Unterscheidet echte vs. generierte Lippen-Paare
```

### Loss Functions
```python
# 1. Adversarial Loss (GAN)
L_adv = -log(D(G(x, c)))

# 2. Reconstruction Loss (L1)
L_rec = ||y - G(x, c)||₁

# 3. Perceptual Loss (VGG Features)
L_perc = ||VGG(y) - VGG(G(x, c))||₂

# 4. Landmark Consistency Loss
L_landmark = ||landmarks(y) - landmarks(G(x, c))||₂

# Total Loss
L_total = λ₁L_adv + λ₂L_rec + λ₃L_perc + λ₄L_landmark
```

### Performance Targets
- **Generation Time:** < 2 Sekunden
- **Image Quality:** SSIM > 0.85, LPIPS < 0.15
- **Landmark Accuracy:** < 2px deviation
- **User Satisfaction:** > 80% "realistisch" Bewertung

## 📱 MVP User Experience

### Input
- Upload 1-3 Bilder (Frontal, Links, Rechts)
- Auswahl: Produkt (Restylane Kysse, etc.)
- Eingabe: Volumen (0.5-2.0ml)

### Output  
- Vorher/Nachher Vergleich
- 3D Ansicht (optional)
- Confidence Score
- "Dies ist eine Simulation" Disclaimer

## 🧪 Evaluation Strategy

### Quantitative Metriken
- **FID Score** (Fréchet Inception Distance)
- **SSIM** (Structural Similarity)
- **LPIPS** (Learned Perceptual Image Patch Similarity)
- **Landmark Distance Error**

### Qualitative Evaluation
- Expert Review (Ärzte)
- User Study (A/B Tests)
- Realism Score (1-10)

## 🔒 Ethische Überlegungen

### Disclaimer & Warnings
- "Dies ist eine Computersimulation"
- "Echte Ergebnisse können abweichen"
- "Konsultieren Sie einen Arzt"

### Data Privacy
- Keine Speicherung von Nutzerbildern
- GDPR-konform
- Anonymisierte Metriken

## 📈 Skalierungsstrategie

### Phase 1: MVP (50 Personen Dataset)
- Proof of Concept
- Basic Lippenaugmentation

### Phase 2: Erweitert (500+ Personen)
- Mehr Produkte & Volumen
- Verschiedene Ethnizitäten
- Altersgruppen

### Phase 3: Professional (5000+ Personen)
- Real-time Generation
- Mobile App
- Integration in Praxis-Software