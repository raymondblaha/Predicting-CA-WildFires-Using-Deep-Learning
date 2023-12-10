import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, Reshape, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
import joblib


# Load the datasets
train_data = pd.read_csv('/Volumes/LaCie/Deep_Learning_Final_Project/Machine/Training_Final.csv')
val_data = pd.read_csv('/Volumes/LaCie/Deep_Learning_Final_Project/Machine/Vaildation_Final.csv')

# Handle missing values
def handle_nan_values(df):
    # Forward fill
    df.ffill(inplace=True)
    # Backward fill
    df.bfill(inplace=True)
    # Fill remaining NaNs with the mean (or another strategy if more appropriate)
    df.fillna(df.mean(), inplace=True)

handle_nan_values(train_data)
handle_nan_values(val_data)

# One-hot encode categorical data
train_data = pd.get_dummies(train_data, columns=['CountyName', 'Ecosystem'])
val_data = pd.get_dummies(val_data, columns=['CountyName', 'Ecosystem'])

# Align features in training and validation data
train_features = train_data.drop(['date', 'Fire'], axis=1)
val_features = val_data.drop(['date', 'Fire'], axis=1)

# Add missing columns in validation set
missing_cols = set(train_features.columns) - set(val_features.columns)
for c in missing_cols:
    val_features[c] = 0

# Ensure the order of feature columns in validation data is the same as in training data
val_features = val_features[train_features.columns]

# Check for columns with zero variance
zero_variance_cols_train = train_features.columns[train_features.std() == 0]
zero_variance_cols_val = val_features.columns[val_features.std() == 0]
print("Zero variance columns in training data:", zero_variance_cols_train)
print("Zero variance columns in validation data:", zero_variance_cols_val)

# Identify zero-variance columns in training and validation datasets
zero_variance_cols_train = train_features.columns[train_features.std() == 0]
zero_variance_cols_val = val_features.columns[val_features.std() == 0]

# Combine the lists of zero-variance columns and drop them from both datasets
all_zero_variance_cols = set(zero_variance_cols_train) | set(zero_variance_cols_val)
train_features = train_features.drop(columns=all_zero_variance_cols)
val_features = val_features.drop(columns=all_zero_variance_cols)

# Check for NaN values in the features before scaling
if train_features.isna().any().any() or val_features.isna().any().any():
    raise ValueError("NaN values found in features before scaling")

# Save the feature names here, after preprocessing but before scaling
feature_names = train_features.columns
np.save('/Volumes/LaCie/Deep_Learning_Final_Project/Machine/feature_names.npy', feature_names)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
X_val = scaler.transform(val_features)

# Save the scaler
joblib.dump(scaler, '/Volumes/LaCie/Deep_Learning_Final_Project/Machine/scaler_final.gz')

# Undersample the majority class in the training data
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_resampled, Y_train_resampled = undersample.fit_resample(X_train, train_data['Fire'].values)

# Check for any NaN or infinite values after normalization
if np.any(np.isnan(X_train)) or np.any(np.isnan(X_val)):
    raise ValueError("NaN values found in scaled datasets after normalization")
if np.any(np.isinf(X_train)) or np.any(np.isinf(X_val)):
    raise ValueError("Infinite values found in scaled datasets after normalization")

Y_train = train_data['Fire'].values
Y_val = val_data['Fire'].values

# Check for NaN values in the data
print(train_data.isna().sum())
print(val_data.isna().sum())

# Check for extreme values in the data
print(train_data.describe())
print(val_data.describe())


# Define the autoencoder architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
encoded = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(encoded)
decoded = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')  # Use mean squared error as the loss function

# Train the autoencoder
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, validation_data=(X_val, X_val), callbacks=[early_stopping])


# Define the encoder model separately (after training the autoencoder)
encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_1').output)

# Encode and reshape training data for CNN input
encoded_X_train = encoder_model.predict(X_train)
reshaped_dim = int(np.sqrt(64))  # 8x8 grid
reshaped_X_train = encoded_X_train.reshape(-1, reshaped_dim, reshaped_dim, 1)

# Similarly, encode and reshape validation data for CNN input
encoded_X_val = encoder_model.predict(X_val)
reshaped_X_val = encoded_X_val.reshape(-1, reshaped_dim, reshaped_dim, 1)

# Undersample the majority class in the reshaped training data
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_reshaped_resampled, Y_train_resampled = undersample.fit_resample(
    reshaped_X_train.reshape(-1, reshaped_dim * reshaped_dim),  # Flatten for undersampling
    Y_train
)
# Reshape back for CNN input
X_train_reshaped_resampled = X_train_reshaped_resampled.reshape(-1, reshaped_dim, reshaped_dim, 1)


# CNN Model with regularization and increased dropout
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(reshaped_dim, reshaped_dim, 1), kernel_regularizer=l2(0.001)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Dropout(0.5))
cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
cnn.add(Flatten())
cnn.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN with the resampled data
cnn.fit(
    X_train_reshaped_resampled, Y_train_resampled,
    epochs=10,
    batch_size=256,
    validation_data=(reshaped_X_val, Y_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)

# LSTM (to process the original temporal sequences)
# Reshape data for LSTM input
sequence_length = 14  # Example: 14 days sequence
def create_sequences(X, y, sequence_length):
    sequences = []
    labels = []
    for i in range(len(X)):
        if i < sequence_length - 1:
            # Pad sequence with zeros at the beginning
            pad_length = sequence_length - 1 - i
            padded_sequence = np.zeros((sequence_length, X.shape[1]))
            padded_sequence[pad_length:] = X[:i+1]
            sequences.append(padded_sequence)
            labels.append(y[i])
        else:
            seq = X[i - sequence_length + 1:i + 1]
            sequences.append(seq)
            labels.append(y[i])
    return np.array(sequences), np.array(labels)

# Create sequences for both training and validation sets
X_train_seq, Y_train_seq = create_sequences(X_train, Y_train, sequence_length)
X_val_seq, Y_val_seq = create_sequences(X_val, Y_val, sequence_length)


model3 = Sequential()
model3.add(Conv1D(64, 2, activation="relu", input_shape=(sequence_length, input_dim)))
model3.add(Dense(100, activation="relu"))
model3.add(Conv1D(64, 2, activation="relu"))
model3.add(Dense(100, activation="relu"))
model3.add(Conv1D(64, 2, activation="relu"))
model3.add(MaxPooling1D())
model3.add(Flatten())
model3.add(Dense(100, activation="relu"))
model3.add(Dense(100, activation="relu"))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(X_train_seq, Y_train_seq, epochs=10, batch_size=256, validation_data=(X_val_seq, Y_val_seq), callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# Save models
autoencoder.save('/Volumes/LaCie/Deep_Learning_Final_Project/Machine/autoencoder_model_final.h5')
cnn.save('/Volumes/LaCie/Deep_Learning_Final_Project/Machine/cnn_model_final.h5')
model3.save('/Volumes/LaCie/Deep_Learning_Final_Project/Machine/lstm_model_final.h5')

# Predict on validation data using each model
encoded_X_val = encoder_model.predict(X_val)

# Assuming the encoder outputs 64 features, reshape it into an 8x8 grid
reshaped_dim = int(np.sqrt(encoded_X_val.shape[1]))  # Calculate the square root of 64
reshaped_X_val = encoded_X_val.reshape(-1, reshaped_dim, reshaped_dim, 1)

# The rest of the code remains the same
X_val_seq, _ = create_sequences(X_val, np.zeros((X_val.shape[0],)), 14)
predictions_autoencoder = autoencoder.predict(X_val)
predictions_cnn = cnn.predict(reshaped_X_val)
predictions_lstm = model3.predict(X_val_seq)
# Assuming the LSTM model's predictions are in 'predictions_lstm'
lstm_final_predictions = predictions_lstm[:, -1]
# Ensure all predictions are of the same shape
# Assuming you want to use the first feature from the autoencoder's predictions
autoencoder_final_predictions = np.mean(predictions_autoencoder, axis=1)
cnn_final_predictions = predictions_cnn.flatten()


print("Autoencoder Final Predictions Shape:", autoencoder_final_predictions.shape)
print("CNN Predictions Shape:", predictions_cnn.shape)
print("LSTM Final Predictions Shape:", lstm_final_predictions.shape)


# Define a threshold value
threshold = 0.3  # for example, to be more sensitive to fires

# Combine the predictions
predictions_ensemble = (autoencoder_final_predictions + cnn_final_predictions + lstm_final_predictions) / 3

# Apply the threshold
ensemble_binary_predictions = (predictions_ensemble > threshold).astype(int)

# Evaluate the ensemble model
print(classification_report(Y_val, ensemble_binary_predictions))

np.save('/Volumes/LaCie/Deep_Learning_Final_Project/Machine/ensemble_predictions.npy', ensemble_binary_predictions)
