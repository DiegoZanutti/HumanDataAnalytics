from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import load_model
from utils.utils import load_person_df_map


model_name = 'simple_CNN'
# Load the model
model = load_model(f'{model_name}_model.h5')
WALKING = 1
DESCENDING = 2
ASCENDING = 3
DRIVING = 4
activities_list_to_consider = [WALKING, DESCENDING, ASCENDING, DRIVING]
person_df_map = load_person_df_map(activities_list_to_consider)

def predict_activity(person_df_map, person_id, activity_id, model, window_size=128):
    scaler = StandardScaler()

    # Extract the data
    activity_data = person_df_map[person_id]['activities'][activity_id]
    activity_data = activity_data[['lw_x', 'lw_y', 'lw_z', 'lh_x', 'lh_y', 'lh_z', 'la_x', 'la_y', 'la_z', 'ra_x', 'ra_y', 'ra_z']]
    # if the activity id is empty, skip the transformation and prediction
    if activity_data.shape[0] == 0: # which means zero rows
        print(f"Skipping person {person_id} activity {activity_id} due to empty dataframe")
    else:
      # Scale the data
      scaled_data = scaler.fit_transform(activity_data)

      # Segment the data into windows
      segments = []
      for start in range(0, len(scaled_data) - window_size, window_size):
          end = start + window_size
          segments.append(scaled_data[start:end])

      # Reshape the data
      segments = np.array(segments)
      segments = segments.reshape((-1, window_size, segments.shape[2]))

      # Predict
      predictions = model.predict(segments)

      # Aggregate predictions (e.g., by taking the mode)
      predicted_classes = np.argmax(predictions, axis=1)
      most_common_prediction = np.bincount(predicted_classes).argmax()

      return most_common_prediction

activity_names = {WALKING: "Walking", DESCENDING: "Descending Stairs", ASCENDING: "Ascending Stairs", DRIVING: "Driving"}
# predicted_activity = predict_activity(person_df_map, 'id5993bf4a', DESCENDING, model)
# predicted_activity_label = activity_names[predicted_activity+1]
# print(f"Predicted Activity: {predicted_activity_label}")


for person_id in person_df_map.keys():
  for activity_id, expected_activity_label in activity_names.items():
    # return the predicted activity id
    predicted_activity = predict_activity(person_df_map, person_id, activity_id, model)
    if predicted_activity is None:
      print(f"Skipping person {person_id} activity {activity_id} due to empty dataframe")
      continue
    else:
      predicted_activity += 1 
      predicted_activity_label = activity_names[predicted_activity]

      if activity_id != predicted_activity:
        print(f"********** -----> Predicted incorrectly! {person_id}, {predicted_activity_label}, expected: {expected_activity_label}")
      else:
        print(f"Predicted Activity: {predicted_activity_label}")