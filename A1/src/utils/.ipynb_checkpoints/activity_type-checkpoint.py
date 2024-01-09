import numpy as np
class ActivityType:
    WALKING = 1
    DESCENDING_STAIRS = 2
    ASCENDING_STAIRS = 3
    DRIVING = 4
    CLAPPING = 77
    NON_STUDY_ACTIVITY = 99

    @staticmethod
    def get_activity_name(activity_id):
        activity_mapping = {
            ActivityType.WALKING: "Walking",
            ActivityType.DESCENDING_STAIRS: "Descending Stairs",
            ActivityType.ASCENDING_STAIRS: "Ascending Stairs",
            ActivityType.DRIVING: "Driving",
            ActivityType.CLAPPING: "Clapping",
            ActivityType.NON_STUDY_ACTIVITY: "Non-Study Activity"
        }
        return activity_mapping.get(activity_id, "Unknown")

    def create_label_mapping():
        activity_ids = [ActivityType.WALKING, ActivityType.DESCENDING_STAIRS, 
                        ActivityType.ASCENDING_STAIRS]
        label_mapping = {label: idx for idx, label in enumerate(activity_ids)}
        return label_mapping

    def one_hot(y_, label_mapping):
        """
        Function to encode output labels from number indexes to one-hot encoding.
        """
        mapped_labels = np.vectorize(label_mapping.get)(y_)
        n_values = len(label_mapping)
        return np.eye(n_values)[mapped_labels.astype(np.int32)]


