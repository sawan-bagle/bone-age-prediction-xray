# Import necessary libraries 
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image 
import tensorflow as tf 
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.preprocessing.image import load_img, 
img_to_array, ImageDataGenerator 
from  tensorflow.keras.optimizers  import  Adam 
from tensorflow.keras.callbacks import EarlyStopping  
from sklearn.metrics import mean_absolute_error, mean_squared_error, 
r2_score 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
import keras_tuner as kt  
# Suppress warnings 
import warnings 
warnings.filterwarnings("ignore") 
print("All libraries imported successfully!") 
 
path_to_csv = "." 
path_to_images = "." 
 
train_csv_path = os.path.join(path_to_csv, "boneage-training- 
dataset.csv") 
# test_csv_path = os.path.join(path_to_csv, "boneage-test- 
dataset.csv") # We might not need this if splitting train_csv 
train_df  =  pd.read_csv(train_csv_path) 
# test_df = pd.read_csv(test_csv_path) # Load if needed for final 
prediction submission, not for evaluation here 
# Add file paths to the dataframe 
train_df['Image  Path']  =  train_df['id'].astype(str)  +  ".png" 
# test_df['Image Path'] = test_df['Case ID'].astype(str) + ".png" 
 
train_img_path = os.path.join(path_to_images, "boneage-training- 
dataset", "boneage-training-dataset") 
# test_img_path = os.path.join(path_to_images, "boneage-test-dataset", 
"boneage-test-dataset") 
if not os.path.isdir(train_img_path): 
raise FileNotFoundError(f"Training image directory not found at: 
{train_img_path}") 
print("Dataset loaded successfully!") 
print(f"Training image path set to: {train_img_path}") 
train_df.head() 
 
IMG_SIZE = (224, 224) # Define image size constant 
def load_images(df, path, img_size): 
images = [] 
image_paths_full = [os.path.join(path, img_name) for img_name in 
df['Image Path']] 
missing_files = [p for p in image_paths_full if not 
os.path.exists(p)] 
if missing_files: 
print(f"Warning: Missing {len(missing_files)} image files. 
First few:") 
print(missing_files[:5]) 
 
existing_files_set = set(image_paths_full) - 
set(missing_files) 
df_filtered = df[df['Image Path'].apply(lambda x: 
os.path.join(path, x) in existing_files_set)].copy() 
print(f"Proceeding with {len(df_filtered)} images.") 
else: 
df_filtered = df.copy() 
if df_filtered.empty: 
raise ValueError("No valid image paths found after checking 
existence.") 
for img_name in df_filtered['Image Path']: 
try: 
img = load_img(os.path.join(path, img_name), 
target_size=img_size) 
img = img_to_array(img) / 255.0  
images.append(img) 
except Exception as e: 
print(f"Error loading image {img_name}: {e}") 
 
continue 
 
 
y_values  =  df_filtered['boneage'].values 
if len(images) != len(y_values): 
raise ValueError(f"Mismatch between number of loaded images 
({len(images)}) and targets ({len(y_values)}). This might happen if 
some images failed to load.") 
return np.array(images), y_values, df_filtered  
 
 X, y, train_df_filtered = load_images(train_df, train_img_path, 
IMG_SIZE) 
if X.shape[0] == 0: 
raise ValueError("No images were loaded. Check image paths and 
file  integrity.") 
 
# --- Split into Train, Validation, & Test sets --- 
# First split: 80% for train+validation, 20% for test 
X_temp, X_test, y_temp, y_test = train_test_split( 
X, y, test_size=0.2, random_state=42 
) 
# Second split: Split the 80% into 80% train (64% overall) and 20% 
validation (16% overall) 
X_train, X_val, y_train, y_val = train_test_split( 
X_temp, y_temp, test_size=0.2, random_state=42 # 0.2 * 0.8 = 0.16 
) 
# datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, 
horizontal_flip=True) 
# datagen.fit(X_train) 
print("Images loaded and preprocessed successfully!") 
print(f"Train shape: {X_train.shape} {y_train.shape}") 
print(f"Val shape: {X_val.shape} {y_val.shape}") 
print(f"Test shape: {X_test.shape} {y_test.shape}") 
All libraries imported successfully! 
Dataset loaded successfully! 
Training image path set to: .\boneage-training-dataset\boneage- 
training-dataset 
Images loaded and preprocessed successfully! 
Train shape: (8070, 224, 224, 3) (8070,) 
Val shape: (2018, 224, 224, 3) (2018,) 
Test shape: (2523, 224, 224, 3) (2523,) 
 
# --- CNN Model Building Function for Keras Tuner --- 
def build_cnn_model(hp): 
model = Sequential() 
model.add(Conv2D( 
filters=hp.Int('conv_1_filters', min_value=32, max_value=64, 
step=32), 
kernel_size=(3,3), activation='relu', 
input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3) 
)) 
model.add(MaxPooling2D(2,2)) 
model.add(Conv2D( 
filters=hp.Int('conv_2_filters', min_value=64, max_value=128, 
step=32), 
kernel_size=(3,3), activation='relu' 
)) 
model.add(MaxPooling2D(2,2)) 
if hp.Boolean("add_extra_conv_layer"): 
model.add(Conv2D( 
filters=hp.Int('conv_3_filters', min_value=128, 
max_value=256, step=64), 
kernel_size=(3,3), activation='relu' 
)) 
model.add(MaxPooling2D(2,2)) 
model.add(Flatten()) 
model.add(Dense( 
units=hp.Int('dense_units', min_value=256, max_value=512, 
step=128), 
activation='relu' 
)) 
model.add(Dropout( 
rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, 
step=0.1) 
)) 
model.add(Dense(1, activation='linear'))  
model.compile( 
optimizer=Adam(learning_rate=hp.Choice('learning_rate', 
values=[1e-3, 1e-4, 5e-5])), 
loss='mean_squared_error', 
metrics=['mse'] 
) 
return  model 
 
 
tuner_cnn = kt.RandomSearch( 
build_cnn_model, 
objective='val_loss', 
max_trials=10,  
 executions_per_trial=1, 
directory='keras_tuner_cnn', 
project_name='bone_age_cnn' 
) 
tuner_cnn.search_space_summary() 
 
early_stopping = EarlyStopping(monitor='val_loss', patience=3, 
restore_best_weights=True) 
Reloading Tuner from keras_tuner_cnn\bone_age_cnn\tuner0.json 
Search space summary 
Default search space size: 7 
conv_1_filters (Int) 
{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 64, 
'step': 32, 'sampling': 'linear'} 
conv_2_filters (Int) 
{'default': None, 'conditions': [], 'min_value': 64, 'max_value': 128, 
'step': 32, 'sampling': 'linear'} 
add_extra_conv_layer (Boolean) 
{'default': False, 'conditions': []} 
dense_units (Int) 
{'default': None, 'conditions': [], 'min_value': 256, 'max_value': 
512, 'step': 128, 'sampling': 'linear'} 
dropout_1 (Float) 
{'default': 0.1, 'conditions': [], 'min_value': 0.1, 'max_value': 0.5, 
'step': 0.1, 'sampling': 'linear'} 
learning_rate (Choice) 
{'default': 0.001, 'conditions': [], 'values': [0.001, 0.0001, 5e-05], 
'ordered': True} 
conv_3_filters (Int) 
{'default': None, 'conditions': [], 'min_value': 128, 'max_value': 
256, 'step': 64, 'sampling': 'linear'} 
 
 print("Starting CNN hyperparameter search...") 
tuner_cnn.search( 
X_train, y_train, 
epochs=20,  
 validation_data=(X_val, y_val), 
callbacks=[early_stopping], 
batch_size=32  
) 
print("CNN hyperparameter search finished.") 
Trial 10 Complete [00h 41m 18s] 
val_loss: 976.8163452148438 
Best val_loss So Far: 832.185302734375 
Total elapsed time: 09h 04m 12s 
CNN hyperparameter search finished. 
best_hps_cnn = tuner_cnn.get_best_hyperparameters(num_trials=1)[0] 
print(f""" 
Best CNN Hyperparameters Found: 
Conv 1 Filters: {best_hps_cnn.get('conv_1_filters')} 
Conv  2  Filters:  {best_hps_cnn.get('conv_2_filters')} 
Extra Conv Layer: {best_hps_cnn.get('add_extra_conv_layer')} 
Dense Units: {best_hps_cnn.get('dense_units')} 
Dropout Rate: {best_hps_cnn.get('dropout_1')} 
Learning Rate: {best_hps_cnn.get('learning_rate')} 
""")  
 
Best CNN Hyperparameters Found: 
Conv 1 Filters: 64 
Conv 2 Filters: 96 
Extra Conv Layer: True 
Dense Units: 384 
Dropout Rate: 0.30000000000000004 
Learning Rate: 0.001 
 
 
best_cnn_model  =  tuner_cnn.get_best_models(num_models=1)[0] 
 
# best_cnn_model = tuner_cnn.hypermodel.build(best_hps_cnn) 
# history_cnn = best_cnn_model.fit(X_train, y_train, epochs=..., 
validation_data=(X_val, y_val), callbacks=[early_stopping]) 
 
print("Predicting with best CNN model...") 
y_pred_cnn_test  =  best_cnn_model.predict(X_test)  
y_pred_cnn_val  =  best_cnn_model.predict(X_val) 
 
# --- VGG16 Model Building Function for Keras Tuner --- 
 
def build_vgg_model(hp): 
vgg_base = VGG16(weights='imagenet', include_top=False, 
input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)) 
# Freeze VGG base layers 
vgg_base.trainable  =  False 
model = Sequential([ 
vgg_base, 
Flatten(),  
Dense( 
units=hp.Int('dense_units',  min_value=256,  max_value=512, 
step=128), 
activation='relu' 
), 
Dropout( 
rate=hp.Float('dropout_1',  min_value=0.1,  max_value=0.5, 
step=0.1) 
), 
Dense(1, activation='linear') 
]) 
model.compile( 
optimizer=Adam(learning_rate=hp.Choice('learning_rate', 
values=[1e-3, 1e-4, 5e-5])),  
loss='mean_squared_error', 
metrics=['mse'] 
) 
return  model 
Predicting with best CNN model... 
79/79  ━━━━━━━━━━━━━━━━━━━━  8s  104ms/step 
64/64  ━━━━━━━━━━━━━━━━━━━━  7s  114ms/step 
 
tuner_vgg = kt.RandomSearch( 
build_vgg_model, 
objective='val_loss', 
max_trials=5,  
executions_per_trial=1, 
directory='keras_tuner_vgg', 
project_name='bone_age_vgg' 
) 
tuner_vgg.search_space_summary() 
Reloading Tuner from keras_tuner_vgg\bone_age_vgg\tuner0.json 
Search space summary 
Default search space size: 3 
dense_units (Int) 
{'default': None, 'conditions': [], 'min_value': 256, 'max_value': 
512, 'step': 128, 'sampling': 'linear'} 
dropout_1 (Float) 
{'default': 0.1, 'conditions': [], 'min_value': 0.1, 'max_value': 0.5, 
'step': 0.1, 'sampling': 'linear'} 
learning_rate (Choice) 
{'default': 0.001, 'conditions': [], 'values': [0.001, 0.0001, 5e-05], 
'ordered': True} 
 
print("Starting VGG16 hyperparameter search...") 
tuner_vgg.search( 
X_train, y_train, 
epochs=15, # Max epochs per trial 
validation_data=(X_val,  y_val), 
callbacks=[early_stopping],  
batch_size=32  
) 
print("VGG16 hyperparameter search finished.") 
Trial 5 Complete [01h 33m 08s] 
val_loss: 667.1708374023438 
Best val_loss So Far: 544.1021728515625 
Total elapsed time: 10h 00m 39s 
VGG16  hyperparameter  search  finished. 
 
best_hps_vgg = tuner_vgg.get_best_hyperparameters(num_trials=1)[0] 
print(f""" 
Best VGG16 Hyperparameters Found: 
Dense Units: {best_hps_vgg.get('dense_units')} 
Dropout Rate: {best_hps_vgg.get('dropout_1')} 
Learning Rate: {best_hps_vgg.get('learning_rate')} 
""")  
 
Best VGG16 Hyperparameters Found: 
Dense Units: 512 
Dropout Rate: 0.4 
Learning Rate: 0.0001 
 
best_vgg_model  =  tuner_vgg.get_best_models(num_models=1)[0] 
print("Predicting with best VGG16 model...") 
y_pred_vgg_test  =  best_vgg_model.predict(X_test) 
 
y_pred_vgg_val  =  best_vgg_model.predict(X_val) 
Predicting with best VGG16 model... 
79/79 ━━━━━━━━━━━━━━━━━━━━ 91s 1s/step 
64/64 ━━━━━━━━━━━━━━━━━━━━ 73s 1s/step 
 
# --- Evaluate Models on TEST Set--- 
def evaluate_model(y_true, y_pred, model_name): 
r2 = r2_score(y_true, y_pred) 
mae  =  mean_absolute_error(y_true,  y_pred) 
mse  =  mean_squared_error(y_true,  y_pred) 
rmse = np.sqrt(mse) 
print(f"{model_name} Performance (Best Tuned):") 
print(f"R2 Score: {r2:.4f}") 
print(f"MAE: {mae:.4f}") 
print(f"MSE: {mse:.4f}") 
print(f"RMSE: {rmse:.4f}\n") 
return  mse 
print("Evaluating models on the TEST set...\n") 
Evaluating models on the TEST set... 
 
mse_cnn = evaluate_model(y_test, y_pred_cnn_test, "CNN Model") 
mse_vgg = evaluate_model(y_test, y_pred_vgg_test, "VGG16 Model") 
 
print("\nEvaluating Ensembles on the TEST set...\n") 
alpha = mse_vgg / (mse_cnn + mse_vgg)  
print(f"Calculated alpha (weight for CNN in weighted ensemble): 
{alpha:.4f}") 
y_pred_weighted_test = alpha * y_pred_cnn_test + (1 - alpha) * 
y_pred_vgg_test 
mse_weighted = evaluate_model(y_test, y_pred_weighted_test, "Weighted 
Ensemble") 
CNN Model Performance (Best Tuned): R2 Score: 0.4414 
MAE: 24.4425 
MSE: 989.4881 
RMSE: 31.4561 
 
 
 
 
 
VGG16 Model Performance (Best Tuned):  
R2 Score: 0.7063 
MAE: 17.4964 
MSE: 520.2881 
RMSE: 22.8098 
Evaluating Ensembles on the TEST set... 
Calculated alpha (weight for CNN in weighted ensemble): 0.4048 
Weighted Ensemble Performance (Best Tuned): 
R2 Score: 0.6207 
MAE: 20.0599 
MSE: 671.8065 
RMSE: 25.9192 
# Stacking Ensemble 
X_stack_train  =  np.column_stack((y_pred_cnn_val,  y_pred_vgg_val)) 
stack_model = LinearRegression() 
print("Training stacking meta-model on validation set predictions...") 
stack_model.fit(X_stack_train, y_val) 
Training stacking meta-model on validation set predictions... 
LinearRegression() 
X_stack_test = np.column_stack((y_pred_cnn_test, y_pred_vgg_test)) 
print("Predicting with stacking ensemble on test set predictions...") 
y_pred_stack_test  =  stack_model.predict(X_stack_test) 
mse_stack = evaluate_model(y_test, y_pred_stack_test, "Stacking 
Ensemble") 
Predicting with stacking ensemble on test set predictions... 
Stacking Ensemble Performance (Best Tuned): 
R2 Score: 0.7233 
MAE: 16.6574 
MSE: 490.0486 
RMSE: 22.1370 
# Choose best model based on TEST set MSE 
best_model_name = min( 
("CNN", mse_cnn), 
("VGG16", mse_vgg), 
("Weighted Ensemble", mse_weighted), 
("Stacking Ensemble", mse_stack), 
key=lambda x: x[1] 
)[0] 
print(f"Best Performing Model: {best_model_name}") 
Best Performing Model: Stacking Ensemble 
# --- Visualization of Best Model --- 
plt.figure(figsize=(10, 5)) 
 
if best_model_name == "Stacking Ensemble": 
y_pred_best_test = y_pred_stack_test 
elif best_model_name == "Weighted Ensemble": 
y_pred_best_test = y_pred_weighted_test 
elif best_model_name == "CNN": 
y_pred_best_test = y_pred_cnn_test 
else:  
y_pred_best_test  =  y_pred_vgg_test 
plt.scatter(y_test, y_pred_best_test, alpha=0.5, 
label=best_model_name) 
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', 
linestyle='--')  
plt.xlabel("Actual Bone Age (Test Set)") 
plt.ylabel("Predicted  Bone  Age  (Test  Set)") 
plt.title(f"Actual vs Predicted - {best_model_name} (Tuned)") 
plt.legend() 
plt.grid(True) 
plt.show() 
 
 
 y_pred_all_test = [y_pred_cnn_test, y_pred_vgg_test, 
y_pred_weighted_test,  y_pred_stack_test] 
model_names = ["CNN (Tuned)", "VGG16 (Tuned)", "Weighted Ens.", 
"Stacking Ens."] 
r2_scores_test = [r2_score(y_test, y_pred) for y_pred in 
y_pred_all_test] 
mae_scores_test = [mean_absolute_error(y_test, y_pred) for y_pred in 
y_pred_all_test] 
mse_scores_test = [mean_squared_error(y_test, y_pred) for y_pred in 
y_pred_all_test] 
rmse_scores_test = [np.sqrt(mse) for mse in mse_scores_test] 
 
def plot_metric_comparison(metric_values, metric_name, model_labels): 
plt.figure(figsize=(10, 6)) 
bars = plt.bar(model_labels, metric_values, color=['#1f77b4', 
'#ff7f0e', '#2ca02c', '#d62728'], width=0.5)  
plt.xlabel("Models") 
plt.ylabel(metric_name) 
plt.title(f"{metric_name} Comparison Across Models ") 
plt.xticks(rotation=15, ha='right')  
 
 
for bar in bars: 
yval = bar.get_height() 
plt.text(bar.get_x()  +  bar.get_width()/2.0,  yval, 
f"{yval:.4f}", va='bottom', ha='center', fontsize=10)  
plt.tight_layout()   
plt.show() 
plot_metric_comparison(r2_scores_test, "R2 Score", model_names) 
plot_metric_comparison(mae_scores_test, "Mean Absolute Error (MAE)", 
model_names) 
plot_metric_comparison(mse_scores_test, "Mean Squared Error (MSE)", 
model_names) 
plot_metric_comparison(rmse_scores_test, "Root Mean Squared Error 
(RMSE)", model_names) 
