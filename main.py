import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ✅ Step 1: Generate synthetic industrial sensor data
print("✅ Generating industrial sensor data...")

np.random.seed(42)
time_steps = 1000

# Simulate normal sensor readings (temperature, vibration, pressure)
normal_data = pd.DataFrame({
    'temperature': 70 + 5 * np.sin(np.linspace(0, 20, time_steps)) + np.random.normal(0, 0.5, time_steps),
    'vibration': 2 + 0.5 * np.sin(np.linspace(0, 30, time_steps)) + np.random.normal(0, 0.1, time_steps),
    'pressure': 100 + 10 * np.sin(np.linspace(0, 15, time_steps)) + np.random.normal(0, 1, time_steps),
})

# Inject anomalies into test data
anomaly_data = normal_data.copy()
anomaly_indices = [200, 201, 202, 400, 401, 600, 601, 602, 603, 800]
for idx in anomaly_indices:
    anomaly_data.loc[idx, 'temperature'] += np.random.uniform(20, 40)
    anomaly_data.loc[idx, 'vibration'] += np.random.uniform(5, 10)
    anomaly_data.loc[idx, 'pressure'] += np.random.uniform(30, 50)

print(f"✅ Normal samples: {len(normal_data)}")
print(f"✅ Anomaly samples injected at: {anomaly_indices}")

# ✅ Step 2: Preprocess data
scaler = MinMaxScaler()
normal_scaled = scaler.fit_transform(normal_data)
anomaly_scaled = scaler.transform(anomaly_data)

# Create sequences
SEQ_LEN = 20

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

train_sequences = create_sequences(normal_scaled, SEQ_LEN)
test_sequences = create_sequences(anomaly_scaled, SEQ_LEN)

# Convert to tensors
train_tensor = torch.FloatTensor(train_sequences)
test_tensor = torch.FloatTensor(test_sequences)

print(f"✅ Train sequences shape: {train_tensor.shape}")
print(f"✅ Test sequences shape: {test_tensor.shape}")

# ✅ Step 3: Build LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, x):
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Use last hidden state as context
        context = encoded[:, -1:, :].repeat(1, x.shape[1], 1)
        
        # Decode
        decoded, _ = self.decoder(context)
        return decoded

# Initialize model
INPUT_SIZE = 3   # temperature, vibration, pressure
HIDDEN_SIZE = 64
NUM_LAYERS = 2

model = LSTMAutoencoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
print(f"\n✅ Model Architecture:\n{model}")

# ✅ Step 4: Train the model
print("\n✅ Training started...")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
BATCH_SIZE = 32
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    # Mini-batch training
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
    reconstruction_errors = torch.mean((test_tensor - reconstructed) ** 2, dim=(1, 2)).numpy()

# Set threshold (mean + 2 std of training reconstruction errors)
with torch.no_grad():
    train_reconstructed = model(train_tensor)
    train_errors = torch.mean((train_tensor - train_reconstructed) ** 2, dim=(1, 2)).numpy()

threshold = np.mean(train_errors) + 2 * np.std(train_errors)
anomalies_detected = reconstruction_errors > threshold

print(f"✅ Threshold: {threshold:.6f}")
print(f"✅ Anomalies detected: {anomalies_detected.sum()} out of {len(anomalies_detected)} sequences")

# ✅ Step 6: Save the model
torch.save(model.state_dict(), 'anomaly_model.pt')
print("✅ Model saved as anomaly_model.pt")

# ✅ Step 7: Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Training loss
plt.subplot(3, 1, 1)
plt.plot(train_losses)
plt.title('Anomaly Detection - Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

# Plot 2: Reconstruction error with threshold
plt.subplot(3, 1, 2)
plt.plot(reconstruction_errors, label='Reconstruction Error', color='blue')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.title('Sensor Anomaly Detection - Reconstruction Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

# Plot 3: Anomalies highlighted
plt.subplot(3, 1, 3)
plt.plot(reconstruction_errors, label='Reconstruction Error', color='blue', alpha=0.6)
plt.scatter(
    np.where(anomalies_detected)[0],
    reconstruction_errors[anomalies_detected],
    color='red', label='Anomaly Detected', zorder=5, s=50
)
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.title('Sensor Anomaly Detection - Flagged Anomalies')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('anomaly_detection_results.png')
plt.show()
print("✅ Results saved as anomaly_detection_results.png")