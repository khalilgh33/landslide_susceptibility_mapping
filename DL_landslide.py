import os
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ======================
# 1. Load and preprocess vector data
# ======================
os.chdir(r"C:\Users\PC\Downloads\data_landslide")
data = gpd.read_file('landslide_data.shp')
data_array = np.array(data)

# Features (X) and labels (Y)
X = data_array[:, 1:11].astype(float)
Y = data_array[:, 0].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for 1D CNN
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# ======================
# 2. Build and train the 1D CNN model
# ======================
model = Sequential()
model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_data=(X_test_scaled, y_test))

# ======================
# 3. Evaluate the model
# ======================
y_pred = model.predict(X_test_scaled).flatten()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================
# 4. Predict on raster data
# ======================
landslide = rasterio.open('landslide_co11.tif')
landslide_array = landslide.read()
raster_flat = landslide_array.reshape(landslide_array.shape[0], -1).T

# Handle NaNs and invalid values
raster_flat = np.nan_to_num(raster_flat)
raster_flat[raster_flat == -3.4028235e+38] = 0

# Scale and reshape
raster_scaled = scaler.transform(raster_flat)
raster_scaled = raster_scaled.reshape((raster_scaled.shape[0], raster_scaled.shape[1], 1))

# Predict
raster_pred = model.predict(raster_scaled).flatten()
prediction_image = raster_pred.reshape((landslide_array.shape[1], landslide_array.shape[2]))

# ======================
# 5. Save prediction as GeoTIFF
# ======================
output_file = r"C:\Users\Khalil\Desktop\Landslide\data_landslide\cnn_model_prediction.tif"

profile = {
    'driver': 'GTiff',
    'width': prediction_image.shape[1],
    'height': prediction_image.shape[0],
    'count': 1,
    'dtype': 'float32',
    'crs': landslide.crs,
    'transform': landslide.transform,
    'nodata': 0
}

with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(prediction_image.astype(rasterio.float32), 1)

print("✅ CNN prediction raster saved to:", output_file)
