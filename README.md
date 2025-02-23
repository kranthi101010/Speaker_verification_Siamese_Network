# Speaker_verification_Siamese_Network
Speaker Verification Using a Siamese Network with GRU on Short-Time Fourier Transform (STFT) Features
# Speaker Verification using Siamese Network with GRU

## 🚀 Overview  
This project develops a **Speaker Verification System** using a **Siamese Network with Gated Recurrent Units (GRU)**. It determines whether two speech samples belong to the same speaker by analyzing **spectrogram features**.

---

## 🎯 Project Goals  
- **Convert raw speech signals to spectrograms** using **Short-Time Fourier Transform (STFT)**.  
- **Train a Siamese Network on positive & negative speaker pairs** for verification.  
- **Use GRU layers to process time-series spectral features** from audio data.  
- **Evaluate model accuracy and fine-tune for speaker discrimination.**  

---
What I am  Tried to Prove:
✅ A Siamese Network can effectively differentiate speakers using audio spectrograms.
✅ GRU-based feature extraction captures speaker characteristics better than simple CNNs.
✅ Pairwise learning (positive & negative samples) enhances verification accuracy.

---

## 📂 Dataset  
- **Training & Testing Data:** Pre-processed speaker recordings stored in `trs.pkl` and `tes.pkl`.  
- **Each speaker has multiple utterances**, allowing pairwise training.  
- **STFT is applied to transform speech into frequency-time domain representations.**  

---

## 🏗️ Implemented Steps  

### **1️⃣ Data Preprocessing**
✅ **Load speaker recordings** from `.pkl` files.  
✅ **Apply Short-Time Fourier Transform (STFT)** to generate spectrograms.  
✅ **Create positive (same speaker) & negative (different speaker) sample pairs** for training.  

### **2️⃣ Building a Siamese Network**
✅ **GRU-based feature extraction** to process time-series spectrogram inputs.  
✅ **Dot product layer** to measure similarity between feature embeddings.  
✅ **Final classification layer** with **sigmoid activation** for verification.  

### **3️⃣ Model Training & Evaluation**
✅ **Train the model using contrastive learning** with binary cross-entropy loss.  
✅ **Evaluate accuracy on unseen test samples**.  
✅ **Achieved speaker verification accuracy of ~74%**.  

---

## 🏗️ Model Architecture  

**Siamese Network Structure:**  
- **Input:** Two spectrograms (from two speech samples).  
- **Feature Extractor:** GRU layers encode speaker characteristics.  
- **Similarity Score:** Dot product between embeddings.  
- **Output Layer:** Sigmoid activation (1 = same speaker, 0 = different speakers).  

```python
def create_siamese_network_with_gru(input_shape, gru_units=128):
    input1 = tf.keras.Input(shape=input_shape)
    input2 = tf.keras.Input(shape=input_shape)

    # Feature extraction via GRU
    output_1 = tf.keras.layers.GRU(128)(input1)
    output_2 = tf.keras.layers.GRU(128)(input2)

    # Compute similarity
    similarity = tf.keras.layers.Dot(axes=-1, normalize=True)([output_1, output_2])

    # Output layer for binary classification
    output = tf.keras.layers.Dense(1, activation="sigmoid")(similarity)

    return tf.keras.Model(inputs=[input1, input2], outputs=output)
```

📊 Results & Performance

Model	Test Accuracy (%)   of Siamese GRU Network	is 74%
📈 Accuracy improves as more diverse speaker pairs are used during training.

