import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.svm import SVR

# Step 1: Load and Preprocess Data
data = pd.read_csv('DelhiAQI.csv')  # Load the dataset

# Assuming 'AQI' is the target variable and the rest are features
X = data.drop(columns=['AQI'])
y = data['AQI']

# Normalize the features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize the target variable
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Proceed with PCA, SVR, and model evaluation


def build_model_1(input_shape):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform')(inputs)
    features = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(1, activation='relu', kernel_initializer='he_uniform')(features)
    
    # Create the main model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    
    # Model to extract features from the second layer
    feature_extractor = tf.keras.Model(inputs=inputs, outputs=features)
    
    return model, feature_extractor


def build_model_2(input_shape):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform')(inputs)
    features = tf.keras.layers.Dense(8, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
    outputs = tf.keras.layers.Dense(1, activation='relu', kernel_initializer='he_uniform')(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    
    # Model to extract features from the second layer
    feature_extractor = tf.keras.Model(inputs=inputs, outputs=features)
    
    return model, feature_extractor

# Build the models and feature extractors
model_1, feature_extractor_1 = build_model_1(X_train.shape[1])
model_2, feature_extractor_2 = build_model_2(X_train.shape[1])

# Train model_1
history_1 = model_1.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)


model_1_accuracies = []
for i in range(5):
    test_predictions = model_1.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    model_1_accuracies.append(test_mse)

mean_accuracy_model_1 = np.mean(model_1_accuracies)
print(f"Mean MSE for Model 1 over 5 tests: {mean_accuracy_model_1}")

features_model_1 = feature_extractor_1.predict(X_test)

# Define input shape as the number of features in X_train
input_shape = X_train.shape[1]

# Build the second model and its feature extractor
model_2, feature_extractor_2 = build_model_2(input_shape)

# Train model_2
history_2 = model_2.fit(
    X_train, y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_val, y_val),
    verbose=1
)

model_2_accuracies = []
for i in range(5):
    test_predictions_2 = model_2.predict(X_test)
    test_mse_2 = mean_squared_error(y_test, test_predictions_2)
    model_2_accuracies.append(test_mse_2)

mean_accuracy_model_2 = np.mean(model_2_accuracies)
print(f"Mean MSE for Model 2 over 5 tests: {mean_accuracy_model_2}")

features_model_2 = feature_extractor_2.predict(X_test)

combined_features_test = np.hstack((features_model_1, features_model_2))

features_model_1_train = feature_extractor_1.predict(X_train)
features_model_2_train = feature_extractor_2.predict(X_train)
combined_features_train = np.hstack((features_model_1_train, features_model_2_train))

svr_model = SVR(kernel='rbf')  # You can adjust the kernel as needed
svr_model.fit(combined_features_train, y_train.ravel())  # Flatten the target variable

# Predict using the trained SVR model on the test set
svr_predictions = svr_model.predict(combined_features_test)

svr_mse = mean_squared_error(y_test, svr_predictions)

print(svr_mse)



# Define the number of components to test
components_list = [8, 10, 12]
best_mse = float('inf')
best_components = None

for n_components in components_list:
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(combined_features_train)
    X_test_pca = pca.transform(combined_features_test)
    
    # Train SVR model
    svr = SVR()
    svr.fit(X_train_pca, y_train)
    
    # Make predictions
    y_pred = svr.predict(X_test_pca)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    
    print(f'MSE for {n_components} components: {mse}')
    
    # Check if this is the best MSE
    if mse < best_mse:
        best_mse = mse
        best_components = n_components

print(f'The best number of components is {best_components} with an MSE of {best_mse}')

#The best model seems to be training svr with combination of features extracted from moel1 and model2 and reducing the
#dimension to 10 , then modelling it using svr.


