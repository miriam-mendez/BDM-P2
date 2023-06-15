from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import StringIndexer
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

criteria = 'price'

# Calculate the average using a subquery
partition_cols = ['neighborhood', 'propertyType', 'floor', 'status', 'size', 'rooms', 'bathrooms']
avg_window = Window.partitionBy(*partition_cols).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
df_filtered = df.withColumn('avg', avg(col(criteria)).over(avg_window))

# One-hot encoding for categorical columns
categorical_cols = ['neighborhood', 'propertyType', 'floor', 'status']
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded") for col in categorical_cols]

# Combine all feature columns into a single vector column
feature_cols = ['size', 'rooms', 'bathrooms'] + [col+"_encoded" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Create a Random Forest regressor
rf = RandomForestRegressor(featuresCol='features', labelCol='price')

# Crseate a pipeline with the transformation and Random Forest regressor
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# Split the filtered DataFrame into training and validation sets with a constant seed
(trainingData, validationData) = df_filtered.randomSplit([0.8, 0.2], seed=1310)

# Train the Random Forest model
model = pipeline.fit(trainingData)

# Save the trained model
model.write().overwrite().save("/tmp/pycharm_project_743/data/model/")

# Make predictions on the validation set
predictions = model.transform(validationData)

# Evaluate the model's performance on the validation set
evaluator = RegressionEvaluator(labelCol='price', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print("RMSE on validation data:", rmse)

evaluator_r2 = RegressionEvaluator(labelCol='price', metricName='r2')
r2_score = evaluator_r2.evaluate(predictions)
print("R2 score on validation data:", r2_score)

evaluator_mae = RegressionEvaluator(labelCol='price', metricName='mae')
mae = evaluator_mae.evaluate(predictions)
print("MAE on validation data:", mae)

print()
# Print all values include prediction
predictions.select('neighborhood', 'district', 'date', 'size', 'rooms', 'bathrooms', 'propertyType', 'floor', 'status', 'hasLift', 'price', 'prediction').show(1, truncate=False)