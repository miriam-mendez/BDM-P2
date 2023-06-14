from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import col, avg
from uploadToExploitationZone import getIdealistaRDD
from pyspark.sql import SparkSession
from hdfs import InsecureClient

hdfs_cli = InsecureClient("http://10.4.41.46:9870", user="bdm")
subdir = '/user/bdm/formatted_zone/idealista'
files = [f"{subdir}/{name}" for name in hdfs_cli.list('/user/bdm/formatted_zone/idealista')]

def startSpark():
    """
    start spark session
    """
    spark = SparkSession.builder \
        .appName("Read Parquet Files") \
        .getOrCreate()
    spark.conf.set("spark.sql.parser.escapedStringLiterals", "true")
    return spark

# Convert RDD to DataFrame
dataRDD = getIdealistaRDD(startSpark(), files)
df = dataRDD.toDF()

# Preprocessing: Removing outliers
# Assuming you want to remove outliers based on the 'price' and 'size' columns
# threshold = 0.1
criteria = 'price'

# Calculate the average using a subquery
partition_cols = 'neighborhood' #['neighborhood', 'propertyType', 'floor', 'status', 'size', 'rooms', 'bathrooms']
avg_window = Window.partitionBy(partition_cols).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
df_filtered = df.withColumn('avg', avg(col(criteria)).over(avg_window))
# df_filtered = df_filtered.filter(f'abs({criteria} - avg) <= {threshold}')
# print(df_filtered.count())

# One-hot encoding for categorical columns
categorical_cols = ['neighborhood', 'propertyType', 'floor', 'status']
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded") for col in categorical_cols]

# Combine all feature columns into a single vector column
feature_cols = ['size', 'rooms', 'bathrooms'] + [col+"_encoded" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Create a Random Forest regressor
rf = RandomForestRegressor(featuresCol='features', labelCol='price')

# Create a pipeline with the transformations and Random Forest regressor
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# Split the filtered DataFrame into training and validation sets
(trainingData, validationData) = df_filtered.randomSplit([0.8, 0.2],*partition_cols)

# Train the Random Forest model
model = pipeline.fit(trainingData)

# Save the trained model
# model.save()
model.write().overwrite().save("/tmp/pycharm_project_743/data/model/")

# Make predictions on the validation set
predictions = model.transform(validationData)

# Evaluate the model's performance on the validation set
evaluator = RegressionEvaluator(labelCol='price', metricName='rmse')
rmse = evaluator.evaluate(predictions)

# Print the RMSE (Root Mean Squared Error)
print("RMSE on validation data:", rmse)

# Make predictions on the validation set
predictions = model.transform(validationData)

# Select a few columns to display
display_cols = feature_cols + ['price', 'prediction']

# Show an example of the validation data with predicted price
print("Example of validation data with predicted price:")
predictions.select(display_cols).show(1, truncate=False)