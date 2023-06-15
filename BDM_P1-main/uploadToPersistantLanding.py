from hdfs import InsecureClient
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import io
import os
import datetime
from dotenv import load_dotenv

load_dotenv()
# HDFS paths
temp_dir_name = os.getenv("TEMPORAL_LANDING")
perm_dir_name = os.getenv("PERSISTENT_LANDING")

host= "http://"+os.getenv("MACHINE_IP")+":"+os.getenv("MACHINE_PORT")
user_name=os.getenv("USER_NAME")
# Connect to HDFS using InsecureClient
hdfs_cli = InsecureClient(host, user=user_name)

def get_timestamp():
    return int(datetime.datetime.utcnow().timestamp())

def delete_hdfs_directory(dir_name):
    if not hdfs_cli.status(dir_name, strict=False):
        print(f"The folder {dir_name} does not exist.")
        return
    for root, dirs, files in hdfs_cli.walk(dir_name):
        for file in files:
            hdfs_cli.delete(os.path.join(root, file))
    hdfs_cli.delete(dir_name, recursive=True)
    print(f"The folder {dir_name} was deleted.")

def idealista_schema(json_data):
    parking_present = False
    neighborhood_present = False
    if 'parkingSpace' in json_data.columns:
        parking_present = True
    if 'neighborhood' in json_data.columns:
        neighborhood_present = True
    if len(json_data) >0:
        fields = [
            pa.field('propertyCode', pa.int32()),
            pa.field('thumbnail', pa.string()),
            pa.field('externalReference', pa.string()),
            pa.field('numPhotos', pa.int16()),
            pa.field('floor', pa.string()),
            pa.field('price', pa.int32()),
            pa.field('propertyType', pa.string()),
            pa.field('operation', pa.string()),
            pa.field('size', pa.float32()),
            pa.field('exterior', pa.bool_()),
            pa.field('rooms', pa.int8()),
            pa.field('bathrooms', pa.int8()),
            pa.field('address', pa.string()),
            pa.field('province', pa.string()),
            pa.field('municipality', pa.string()),
            pa.field('district', pa.string()),
            pa.field('country', pa.string()),
            pa.field('latitude', pa.float32()),
            pa.field('longitude', pa.float32()),
            pa.field('showAddress', pa.bool_()),
            pa.field('url', pa.string()),
            pa.field('distance', pa.int16()),
            pa.field('hasVideo', pa.bool_()),
            pa.field('status', pa.string()),
            pa.field('newDevelopment', pa.bool_()),
            pa.field('hasLift', pa.bool_()),
            pa.field('priceByArea', pa.float32()),
            pa.field('detailedType', pa.struct([pa.field('typology', pa.string()),
                                                pa.field('subTypology', pa.string())])),
            pa.field('suggestedTexts', pa.struct([pa.field('subtitle', pa.string()),
                                                pa.field('title', pa.string())])),
            pa.field('hasPlan', pa.bool_()),
            pa.field('has3DTour', pa.bool_()),
            pa.field('has360', pa.bool_()),
            pa.field('hasStaging', pa.bool_()),
            pa.field('topNewDevelopment', pa.bool_())
        ]

        if parking_present:
            fields.append(pa.field('parkingSpace', pa.struct(
                [pa.field('hasParkingSpace', pa.bool_()), pa.field('isParkingSpaceIncludedInPrice', pa.bool_()),
                pa.field('parkingSpacePrice', pa.float32())])))
        if neighborhood_present:
            fields.append(pa.field('neighborhood', pa.string()))
        schema = pa.schema(fields)
    else:
        schema= pa.schema([])
    return schema

def open_data_income_schema():

    pa_schema = pa.schema([
            ("Any", pa.uint16()),
            ("Codi_Districte", pa.uint8()),
            ("Nom_Districte", pa.string()),
            ("Codi_Barri", pa.uint8()),
            ("Nom_Barri", pa.string()),
            ("Població", pa.uint32()),
            ("Index_RFD_Barcelona_100", pa.string()),
    ])
    return pa_schema


def open_data_rent_schema():
    fields=[
        pa.field('Any', pa.uint16()),
        pa.field("Trimestre", pa.uint8()),
        pa.field("Codi_Districte", pa.uint8()),
        pa.field("Nom_Districte", pa.string()),
        pa.field("Codi_Barri", pa.uint8()),
        pa.field("Nom_Barri", pa.string()),
        pa.field("Lloguer_mitja", pa.string()),
        pa.field("Preu", pa.string())
    ]
    schema = pa.schema(fields)

    return schema


def lookup_schema():
    pa_schema = pa.schema([
            ("district", pa.string()),
            ("neighborhood", pa.string()),
            ("district_n_reconciled", pa.string()),
            ("district_n", pa.string()),
            ("district_id", pa.string()),
            ("neighborhood_n_reconciled", pa.string()),
            ("neighborhood_n", pa.string()),
            ("neighborhood_id", pa.string()),
        ])
    return pa_schema


delete_hdfs_directory(perm_dir_name)
print("Creating persistant_landing.....")

# Get list of subdirectories in temporary HDFS directory
subdirs = [f"{temp_dir_name}/{name}" for name in hdfs_cli.list(temp_dir_name) if hdfs_cli.status(f"{temp_dir_name}/{name}")['type'] == 'DIRECTORY']
# Loop over subdirectories and convert files to Parquet
for subdir in subdirs:
    # Get list of files in subdirectory
    timestamp = get_timestamp()
    files = [f"{subdir}/{name}" for name in hdfs_cli.list(subdir)]
    
    # Create corresponding subdirectory in new empty HDFS directory
    perm_subdir = f"{perm_dir_name}/{subdir[len(temp_dir_name):]}"
    hdfs_cli.makedirs(perm_subdir, permission=777)
    count = 0
    for file in files:
        # Check if file is in json format
        count = count + 1
        print("Uploading file: ", file)
        if file.endswith('.json'):
            # Read data from file
            with hdfs_cli.read(file) as reader:
                json_data = pd.read_json(io.BytesIO(reader.read()), orient='records')
            
            if subdir.endswith("idealista"):
                if 'floor' in json_data.columns:
                    json_data['floor'] = json_data['floor'].astype(str)
                schema_idealista = idealista_schema(json_data)
                table = pa.Table.from_pandas(json_data, schema=schema_idealista, preserve_index=True)
            else:
                table = pa.Table.from_pandas(json_data)
            # Convert data to Parquet format
            output_file = f"{perm_subdir}/{os.path.basename(file).replace('.json', '')}_{timestamp}.parquet"
            buffer = pa.BufferOutputStream()
            pq.write_table(table, buffer, compression='snappy', use_dictionary=True, version='2.6')
            
            # Write Parquet file buffer to HDFS
            with hdfs_cli.write(output_file) as writer:
                writer.write(buffer.getvalue())
    
        elif file.endswith('.csv'):
            # Read data from CSV file
            with hdfs_cli.read(file) as reader:
                csv_data = pd.read_csv(io.BytesIO(reader.read()))
            
            # Set schema and convert data to Parquet format
            if subdir.endswith("lookup_tables"):
                schema_lookup = lookup_schema()
                table = pa.Table.from_pandas(csv_data, schema=schema_lookup, preserve_index=True)
            elif subdir.endswith("opendata-rent"):
                # cast it to string to avoid problems
                if 'Preu' in csv_data.columns:
                    csv_data['Preu'] = csv_data['Preu'].astype(str)
                schema_data_rent = open_data_rent_schema()
                table = pa.Table.from_pandas(csv_data, schema=schema_data_rent, preserve_index=True)
            elif subdir.endswith("opendatabcn-income"):
                # cast it to string to avoid problems
                if 'Índex RFD Barcelona = 100' in csv_data.columns:
                    csv_data['Index_RFD_Barcelona_100'] = csv_data['Índex RFD Barcelona = 100'].astype(str)
                schema_data_income = open_data_income_schema()
                table = pa.Table.from_pandas(csv_data, schema=schema_data_income, preserve_index=True)
            else:
                table = pa.Table.from_pandas(csv_data)
            output_file = f"{perm_subdir}/{os.path.basename(file).replace('.csv', '')}_{timestamp}.parquet"
            buffer = pa.BufferOutputStream()
            pq.write_table(table, buffer, compression='snappy', use_dictionary=True, version='2.6')
            
            # Write Parquet file buffer to HDFS
            with hdfs_cli.write(output_file) as writer:
                writer.write(buffer.getvalue())
    print(f"{count} files where uploaded")
print("\nPersistant zone has been successfully populated.")




