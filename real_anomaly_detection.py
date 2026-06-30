import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ✅ Step 1: Load real NASA SMAP sensor data
print("✅ Loading real NASA SMAP sensor data...")

TRAIN_PATH = r"F:\ABB Projects\Anomaly-Detection\data\archive\data\data\train\P-1.npy"
TEST_PATH = r"F:\ABB Projects\Anomaly-Detection\data\archive\data\data\test\P-1.npy"
LABELS_PATH = r"F:\ABB Projects\Anomaly-Detection\data\archive\labeled_anomalies.csv"

# Load data
train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)
labels_df = pd.read_csv(LABELS_PATH)

print(f"✅ Train data shape: {train_data.shape}")
print(f"✅ Test data shape: {test_data.shape}")

# Get anomaly labels for P-1
p1_labels = labels_df[labels_df['chan_id'] == 'P-1']
print(f"\n✅ Anomaly info for P-1:\n{p1_labels[['chan_id', 'anomaly_sequences', 'num_values']].to_string()}")

# ✅ Step 2: Use first feature only (primary sensor channel)
train_single = train_data[:, 0].reshape(-1, 1)
test_single = test_data[:, 0].reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_single)
test_scaled = scaler.transform(test_single)

# Create sequences
SEQ_LEN = 30

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

train_sequences = create_sequences(train_scaled, SEQ_LEN)
test_sequences = create_sequences(test_scaled, SEQ_LEN)

train_tensor = torch.FloatTensor(train_sequences)
test_tensor = torch.FloatTensor(test_sequences)

print(f"\n✅ Train sequences: {train_tensor.shape}")
print(f"✅ Test sequences: {test_tensor.shape}")

# ✅ Step 3: Build LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        encoded, (hidden, cell) = self.encoder(x)
        context = encoded[:, -1:, :].repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(context)
        return decoded

INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2

model = LSTMAutoencoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
print(f"\n✅ Model Architecture:\n{model}")

# ✅ Step 4: Train
print("\n✅ Training started...")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
BATCH_SIZE = 32
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i in range(0, len(train_tensor), BATCH_SIZE):
        batch = train_tensor[i:i+BATCH_SIZE]
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (len(train_tensor) / BATCH_SIZE)
    train_losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

print("✅ Training complete!")

# ✅ Step 5: Detect anomalies
print("\n✅ Detecting anomalies...")
model.eval()
with torch.no_grad():
    reconstructed = model(test_tensor)
    reconstruction_errors = torch.mean(
        (test_tensor - reconstructed) ** 2, dim=(1, 2)
    ).numpy()

with torch.no_grad():
    train_reconstructed = model(train_tensor)
    train_errors = torch.mean(
        (train_tensor - train_reconstructed) ** 2, dim=(1, 2)
    ).numpy()

threshold = np.mean(train_errors) + 2 * np.std(train_errors)
anomalies_detected = reconstruction_errors > threshold

print(f"✅ Threshold: {threshold:.6f}")
print(f"✅ Anomalies detected: {anomalies_detected.sum()} out of {len(anomalies_detected)} sequences")

# ✅ Step 6: Parse real anomaly labels
anomaly_ranges = []
try:
    import ast
    raw = p1_labels['anomaly_sequences'].values[0]
    anomaly_ranges = ast.literal_eval(raw)
    print(f"✅ Real anomaly ranges from NASA labels: {anomaly_ranges}")
except:
    print("⚠️ Could not parse anomaly labels")

# ✅ Step 7: Save model
torch.save(model.state_dict(), 'real_anomaly_model.pt')
print("✅ Model saved as real_anomaly_model.pt")

# ✅ Step 8: Plot results
plt.figure(figsize=(15, 12))

# Plot 1: Raw sensor data
plt.subplot(4, 1, 1)
plt.plot(test_single, color='blue', alpha=0.7)
plt.title('NASA SMAP P-1 — Raw Sensor Signal (Test Data)')
plt.xlabel('Time Step')
plt.ylabel('Sensor Value')
plt.grid(True)

# Plot 2: Training loss
plt.subplot(4, 1, 2)
plt.plot(train_losses, color='green')
plt.title('LSTM Autoencoder — Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

# Plot 3: Reconstruction error with threshold
plt.subplot(4, 1, 3)
plt.plot(reconstruction_errors, color='blue', label='Reconstruction Error')
plt.axhline(y=threshold, color='red', linestyle='--',
            label=f'Threshold: {threshold:.4f}')
for r in anomaly_ranges:
    plt.axvspan(r[0], r[1], alpha=0.2, color='orange', label='Real Anomaly Zone')
plt.title('Sensor Anomaly Detection — Reconstruction Error vs Real Labels')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

# Plot 4: Flagged anomalies
plt.subplot(4, 1, 4)
plt.plot(reconstruction_errors, color='blue', alpha=0.6, label='Reconstruction Error')
plt.scatter(
    np.where(anomalies_detected)[0],
    reconstruction_errors[anomalies_detected],
    color='red', label='Detected Anomaly', zorder=5, s=30
)
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.title('Sensor Anomaly Detection — Flagged Anomalies')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('real_anomaly_results.png')
plt.show()
print("✅ Results saved as real_anomaly_results.png")