from libraries import *


# Load pipeline and model
pipeline = joblib.load('./FLASK Deploy/models/pipeline.joblib')
model = joblib.load('./FLASK Deploy/models/model.joblib')

# Create a new DataFrame with the correct columns
columns = ['Booking_ID', 'number of adults', 'number of children', 'number of weekend nights',
           'number of week nights', 'type of meal', 'car parking space', 'room type', 'lead time',
           'market segment type', 'repeated', 'P-C', 'P-not-C', 'average price', 'special requests',
           'date of reservation']

new_data = pd.DataFrame(columns=columns)

# Add a new row of data
new_data.loc[0] = ['INN06040', 1, 0, 2, 4, 'Meal Plan 1', 0, 'Room_Type 1', 69, 'Online', 0, 0, 0, 120.0, 0, '6/12/2018']

# Transform the data using the pipeline
transformed_data = pipeline.transform(new_data)

# Print the transformed data
print(transformed_data)
print(model.predict(transformed_data))
