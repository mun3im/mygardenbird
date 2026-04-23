import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.run_functions_eagerly(True)

DATASET_DIR = "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"
SR = 16000
N_MELS = 80
HOP = 512
TARGET_FRAMES = 128   # adjust to match your training input width

# -------------------------------------------------------
# Load EfficientNetB0 and extract embedding layer
# -------------------------------------------------------

base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',  # Use pretrained weights instead of None
    include_top=False,
    input_shape=(N_MELS, TARGET_FRAMES, 3)
)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

embedding_model = tf.keras.Model(inputs=base_model.input, outputs=x)

# If you trained a model, load its weights
# embedding_model.load_weights("your_model_weights.h5")

# -------------------------------------------------------
# Feature extraction
# -------------------------------------------------------

BATCH_SIZE = 64

images = []
labels = []
features = []

for label in sorted(os.listdir(DATASET_DIR)):

    species_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(species_dir):
        continue

    files = [f for f in os.listdir(species_dir) if f.endswith(".wav")]

    for f in tqdm(files, desc=label):

        path = os.path.join(species_dir, f)

        try:
            y, sr = librosa.load(path, sr=SR)

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=N_MELS,
                hop_length=HOP
            )

            mel_db = librosa.power_to_db(mel, ref=np.max)

            if mel_db.shape[1] < TARGET_FRAMES:
                pad = TARGET_FRAMES - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0,0),(0,pad)))
            else:
                mel_db = mel_db[:, :TARGET_FRAMES]

            img = np.stack([mel_db, mel_db, mel_db], axis=-1)

            images.append(img)
            labels.append(label)

            # run batch inference
            if len(images) == BATCH_SIZE:
                batch = np.array(images)
                emb = embedding_model.predict(batch, verbose=0)

                features.extend(emb)

                images = []

        except Exception as e:
            print("Error:", path, e)

# run final partial batch
if len(images) > 0:
    batch = np.array(images)
    emb = embedding_model.predict(batch, verbose=0)
    features.extend(emb)

features = np.array(features)

print("Embedding matrix:", features.shape)

# -------------------------------------------------------
# Standardize features
# -------------------------------------------------------

features = StandardScaler().fit_transform(features)

# -------------------------------------------------------
# t-SNE dimensionality reduction
# -------------------------------------------------------

print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=50,        # Try 5-50, typically sqrt(n_samples)
    learning_rate=500,    # Try 100-1000
    n_iter=2000,          # More iterations for convergence
    init='pca',           # Better initialization
    random_state=42,
    metric='euclidean'    # Try 'cosine' for embeddings
)

embedding = tsne.fit_transform(features)

# -------------------------------------------------------
# Plot
# -------------------------------------------------------

plt.figure(figsize=(10,8))

unique_labels = sorted(set(labels))

for label in unique_labels:

    idx = [i for i,l in enumerate(labels) if l == label]

    plt.scatter(
        embedding[idx,0],
        embedding[idx,1],
        s=6,
        alpha=0.7,
        label=label
    )

plt.title("t-SNE visualization of EfficientNet-B0 embeddings")
plt.xlabel("t-SNE-1")
plt.ylabel("t-SNE-2")

plt.legend(markerscale=3, fontsize=8)

plt.tight_layout()
plt.savefig("eb0_tsne.png", dpi=300)
plt.show()