import sys, os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('classifiers'))))

# Set file path to raw accelerometer data (.CSV file, unit=G's, headers = ['ts','x','y','z'])
raw_data_filepath = 'D:\RESEARCH\lyze-tremor-bradykinesia-PD-master\demo\Test.csv'
# Specify raw data sampling rate (float)
raw_data_sampling_rate = 100.

# Load raw data into Pandas DataFrame
raw_data_df = pd.read_csv(raw_data_filepath)
raw_data_df.head()

# Plot data
from matplotlib import pyplot as plt
raw_data_df.plot()
plt.ylabel('G')
plt.xlabel('Samples')
plt.title('Loaded Raw Data')
plt.show()

from classifiers import constants

from classifiers import resting_tremor_classifier

# Build feature set in 3 second windows for resting tremor classification
tremor_classifier_feature_set = resting_tremor_classifier.build_rest_tremor_classification_feature_set(raw_data_df, raw_data_sampling_rate)
tremor_classifier_feature_set.head()

# Trim calculated features to ones determined from feature selection
tremor_classifier_feature_set = tremor_classifier_feature_set[constants.TREMOR_FEATURE_SELECTION]
tremor_classifier_feature_set.head()

# Initialize model for training (Untrained Model)
resting_tremor_model = resting_tremor_classifier.initialize_model()
resting_tremor_model

from classifiers import hand_movement_classifier

# Compute hand movement predictions in 3 second windows
hand_movement_predictions = hand_movement_classifier.detect_hand_movement(raw_data_df, raw_data_sampling_rate)
hand_movement_predictions

from classifiers import resting_tremor_amplitude_classifier

# Compute rest tremor amplitude in 3 second windows
rest_tremor_amplitude_predictions = resting_tremor_amplitude_classifier.calculate_tremor_amplitude(raw_data_df, raw_data_sampling_rate)
rest_tremor_amplitude_predictions

from classifiers import hand_movement_features

# Calculate features of hand movement in 3 second windows
hand_movement_amplitude_predictions, hand_movement_jerk_predictions = hand_movement_features.calculate_amplitude_and_smoothness_features(raw_data_df, raw_data_sampling_rate)
print ('Hand movement amplitude')
print (hand_movement_amplitude_predictions)
print
print ('Smoothness of hand movement')
print (hand_movement_jerk_predictions)

def generate_dummy_predictions(length):
    '''
    Generate dummy binary predictions for example purposes. 
    :param length: (int) length of predictions to generate
    :return: list of random binary predictions
    '''
    import random
    preds = [0]*(length/2) + [1]*(length/2)
    random.shuffle(preds)
    return preds

# Generate dummy predictions for gait and tremor
gait_predictions = generate_dummy_predictions(len(hand_movement_predictions))
tremor_predictions = generate_dummy_predictions(len(hand_movement_predictions))

# Organize all module predictions into 1 Pandas DataFrame
predictions_df = pd.DataFrame()
predictions_df['hand_movement'] = pd.Series(hand_movement_predictions)
predictions_df['gait'] = pd.Series(gait_predictions)
predictions_df['tremor_constancy'] = pd.Series(tremor_predictions)
predictions_df['tremor_amplitude'] = pd.Series(rest_tremor_amplitude_predictions)
predictions_df['hand_movement_amplitude'] = pd.Series(hand_movement_amplitude_predictions)
predictions_df['hand_movement_jerk'] = pd.Series(hand_movement_jerk_predictions)
predictions_df

from endpoints import filter_classifier_predictions

filtered_predictions_df = filter_classifier_predictions.filter_predictions_by_tree(predictions_df)
filtered_predictions_df

from endpoints import resting_tremor_endpoints

# ---------------- # 
# Tremor Constancy
# ---------------- #
tremor_classifier_predictions = filtered_predictions_df.tremor_classifier_predictions.tolist()

# Compute tremor constancy for given data 
predicted_tremor_constancy = resting_tremor_endpoints.compute_tremor_constancy(tremor_classifier_predictions)
print( 'Predicted Tremor Constancy: ', predicted_tremor_constancy, '%')

# ---------------- # 
# Tremor Amplitude
# ---------------- #
# Filter out all 'NA' from tremor amplitude predictions 
tremor_amplitude_predictions = filter(lambda x: x != 'NA', filtered_predictions_df.tremor_amplitude_predictions)

# Compute 85th percentile of tremor amplitude for given data
predicted_aggregate_tremor_amplitude = resting_tremor_endpoints.compute_aggregate_tremor_amplitude(tremor_amplitude_predictions)
print ('Predicted Aggregate Tremor Amplitude', predicted_aggregate_tremor_amplitude, 'G')

from endpoints import bradykinesia_endpoints

# ----------------------- # 
# Hand Movement Amplitude
# ----------------------- #
# Filter out all 'NA' from hand movement amplitude predictions 
hand_movement_amplitude_predictions = filter(lambda x: x != 'NA', filtered_predictions_df.hand_movement_amplitude)

# Compute aggregate hand movement amplitude for given data (Mean)
aggregate_hand_movement_amplitude = bradykinesia_endpoints.compute_aggregate_hand_movement_amplitude(hand_movement_amplitude_predictions)

# ----------------------- # 
# Hand Movement Smoothness
# ----------------------- #
# Filter out all 'NA' from hand movement smoothness predictions
hand_movement_smoothness_predictions = filter(lambda x: x != 'NA', filtered_predictions_df.hand_movement_jerk)

# Compute aggregate hand movement smoothness for given data (95th percentile)
aggregate_hand_movement_smoothness = bradykinesia_endpoints.compute_aggregate_smoothness_of_hand_movement(hand_movement_smoothness_predictions)

# --------------------------- # 
# Percentage No Hand Movement
# --------------------------- #
hand_movement_predictions = filtered_predictions_df.hand_movement_predictions.tolist()

# Compute percentage of no hand movement 
perc_no_hand_movement = bradykinesia_endpoints.compute_aggregate_percentage_of_no_hand_movement(hand_movement_predictions)

# --------------------------- # 
# Length No Hand Movement Bouts
# --------------------------- #
# Compute aggregate no hand movement bout length (mean)
aggregate_no_hm_bout_length = bradykinesia_endpoints.compute_aggregate_length_of_no_hand_movement_bouts(hand_movement_predictions)

# Print Results
print ('Mean hand amplitude:', aggregate_hand_movement_amplitude, 'G')
print ('95th percentile hand movement smoothness:', aggregate_hand_movement_smoothness)
print ('Percentage no hand movement:', perc_no_hand_movement, '%')
print ('Mean no hand movement bout length:', aggregate_no_hm_bout_length, 'sec')



