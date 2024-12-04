import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Define the number of samples you want to generate
n_samples = 1000

# Generate random synthetic data for the inputs
wall_height = np.random.uniform(2, 10, n_samples)  # Wall Height between 2m and 10m
wall_thickness = np.random.uniform(0.1, 0.5, n_samples)  # Wall Thickness between 0.1m and 0.5m
wall_length = np.random.uniform(3, 15, n_samples)  # Wall Length between 3m and 15m

concrete_grade = np.random.choice([25, 30, 35, 40], size=n_samples)  # Concrete Grades
steel_grade = np.random.choice([415, 500, 550], size=n_samples)  # Steel Grades
reinforcement_ratio = np.random.uniform(0.5, 2.5, n_samples)  # Reinforcement Ratio

axial_load = np.random.uniform(50, 200, n_samples)  # Axial Load between 50 and 200 kN
shear_load = np.random.uniform(10, 50, n_samples)  # Shear Load between 10 and 50 kN
overturning_moment = np.random.uniform(100, 500, n_samples)  # Overturning Moment between 100 and 500 kNm

support_type = np.random.choice([1, 0], size=n_samples)  # 1 for Fixed, 0 for Hinged
loading_type = np.random.choice([1, 0], size=n_samples)  # 1 for Monotonic, 0 for Cyclic

# Generate random output data (Energy Dissipation Capacity)
energy_dissipation_capacity = np.random.uniform(0, 500, n_samples)  # Between 0 and 500 units

# Create the DataFrame
data = pd.DataFrame({
    'wall_height': wall_height,
    'wall_thickness': wall_thickness,
    'wall_length': wall_length,
    'concrete_grade': concrete_grade,
    'steel_grade': steel_grade,
    'reinforcement_ratio': reinforcement_ratio,
    'axial_load': axial_load,
    'shear_load': shear_load,
    'overturning_moment': overturning_moment,
    'support_type': support_type,
    'loading_type': loading_type,
    'energy_dissipation_capacity': energy_dissipation_capacity
})

# Split the data into features (X) and target (y)
X = data.drop(columns=['energy_dissipation_capacity'])
y = data['energy_dissipation_capacity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(11,)),  # 11 input features
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

# Ensure the directory exists for saving the model
save_dir = r"F:\Graphical User Interface\GUI2\saved_model"

# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the trained model in the SavedModel format
model.save(save_dir)
print(f"Model saved at: {save_dir}")

# Evaluate the model
loss = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss}")

# Load the saved model to verify it works correctly
loaded_model = tf.keras.models.load_model(save_dir)
print("Model loaded successfully!")
