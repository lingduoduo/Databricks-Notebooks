# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the data associated with the products we will suggest to users.  This notebook was developed on a Databricks ML 14.2 cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will use descriptive information about products we intend to present to the user to create a searchable set of embeddings. These embeddings will be used to enable a fast and flexible search of our products.
# MAGIC
# MAGIC To perform this work, we must load the data about our products to a database table.  We must then configure a model with which we will convert descriptive information about those products into embeddings. We will then trigger an ongoing workflow that will keep our searchable embeddings, *i.e.* our vector search index, in sync with the table. 

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import datasets # hugging face datasets
datasets.utils.logging.disable_progress_bar() # disable progress bars from hugging face

from sentence_transformers import SentenceTransformer

from databricks.vector_search.client import VectorSearchClient

import mlflow

import pyspark.sql.functions as fn

import pandas as pd
import json

import requests

import time

# COMMAND ----------

# MAGIC %md ##Step 1: Load the Dataset
# MAGIC
# MAGIC The dataset we will be using is the [Red Dot Design Award dataset](https://huggingface.co/datasets/xiyuez/red-dot-design-award-product-description), available through HuggingFace. This dataset contains information on award winning products including descriptive text that we can use for searches.  We will treat this as if this were our set of actual products available to sell to customers:

# COMMAND ----------

# DBTITLE 1,Import the HuggingFace Dataset
# import dataset to hugging face dataset
ds = datasets.load_dataset("xiyuez/red-dot-design-award-product-description") 

print(ds)

# COMMAND ----------

# MAGIC %md HuggingFace makes this dataset available as a dictionary dataset.  We'll persist it as a Delta Lake table as this is more typically how users of Databricks would access product information from within the lakehouse.  
# MAGIC
# MAGIC Please note that we are defining the target table for this data in advance so that we can add an [identity field](https://www.databricks.com/blog/2022/08/08/identity-columns-to-generate-surrogate-keys-are-now-available-in-a-lakehouse-near-you.html) to it.  Creating an id field this way simplifies the creation of unique identifiers for each item in our dataset:

# COMMAND ----------

# DBTITLE 1,Persist as Delta Lake Table
# drop any pre-existing indexes on table
vs_client = VectorSearchClient()
try:
  vs_client.delete_index(f"{config['catalog']}.{config['schema']}.{config['vs index']}")
except:
  print('Ignoring error message associated with vs index deletion ...')
  pass


# create table to hold product info
_ = spark.sql('''
  CREATE OR REPLACE TABLE products (
    id bigint GENERATED ALWAYS AS IDENTITY,
    product string,
    category string,
    description string,
    text string
    )'''
  )

# add product info to table
_  = (
  spark
    .createDataFrame(ds['train']) 
    .select('product','category','description','text') # fields in correct order 
    .write
    .format('delta')
    .mode('append')
    .saveAsTable('products')
  )

# read data from table
products = spark.table('products')

# display table contents
display( products )

# COMMAND ----------

# MAGIC %md It's important to note that the text field contains the concatenated names and descriptions for our products.  This is the field on which we will base our later searches.

# COMMAND ----------

# MAGIC %md ##Step 2: Populate the Vector Store
# MAGIC
# MAGIC To enable our application, we need to convert our product search information to embeddings housed in a searchable vector store.  For this, we will make use of the Databricks integrated vector store.  But to create embeddings understood by the vector store, we need to deploy an embedding model to a Databricks model serving endpoint.

# COMMAND ----------

# MAGIC %md ###Step 2a: Deploy Model to MLFlow Registry
# MAGIC
# MAGIC To enable an efficient search of our product information, we need to convert the descriptive text associated with each record into an embedding.  Each embedding is a numerical representation of the content within the text.  We can use any transformer model capable of converting text to an embedding for this stage.  We are using the [all-MiniLM-L6-v2 model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) because this model produces a reasonably compact embedding (vector) and has been trained for general purpose language scenarios:

# COMMAND ----------

# DBTITLE 1,Download HuggingFace Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# COMMAND ----------

# MAGIC %md To better understand what this model produces, we can ask it to encode some simple strings.  Notice how each one is converted to a floating point array representing an embedding.  These arrays provide a map of the content within each provided unit of text based on the information encoded within the downloaded model:

# COMMAND ----------

# DBTITLE 1,Generate a Sample Embedding
# some sample sentences to convert to embeddings
sentences = [
  "This is an example sentence", 
  "Each sentence is converted into an embedding",
  "An embedding is nothing more than a large, numerical representation of the contents of a unit of text"
  ]

# convert the sentences to embeddings
embeddings = model.encode(sentences)

# display the embeddings
display(embeddings)

# COMMAND ----------

# MAGIC %md Using the sample sentences and the resulting embeddings, we can generate a signature for our model.  A signature is nothing more than a lightweight schema that defines the expected structure of the inputs and outputs for a given model.  This signature will assist us in deploying our model behind a model serving endpoint later in this notebook:

# COMMAND ----------

# DBTITLE 1,Generate a Model Signature
signature = mlflow.models.signature.infer_signature(sentences, embeddings)
print(signature)

# COMMAND ----------

# MAGIC %md We can now register the model along with its signature in the [MLFlow registry](https://docs.databricks.com/en/mlflow/model-registry.html).  This is key step in the publication of the model to a model serving endpoint.  Please note that we are registering this model using the [*sentence_transformers* model flavor](https://mlflow.org/docs/latest/python_api/mlflow.sentence_transformers.html) within MLFlow which is specifically designed to work with this class of model:

# COMMAND ----------

# DBTITLE 1,Register Model with MLFlow
# identify the experiment that will house this mlflow run
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Users/{user_name}/{config['embedding_model_name']}")
             
# initiate mlflow run to log model to experiment
with mlflow.start_run(run_name=config['embedding_model_name']) as run:

  model_info = mlflow.sentence_transformers.log_model(
    model,
    artifact_path='model',
    signature=signature,
    input_example=sentences,
    registered_model_name=config['embedding_model_name']
    )

# COMMAND ----------

# MAGIC %md With our model registered in MLFlow, we can now determine the model version associated with this deployment.  As we deploy subsequent versions of this model, the MLFlow version number will be incremented:

# COMMAND ----------

# DBTITLE 1,Get Registered Model to Version
# connect to mlflow
mlf_client = mlflow.MlflowClient()

# get last version of registered model
model_version = mlf_client.get_latest_versions(config['embedding_model_name'])[0].version

print(model_version)

# COMMAND ----------

# MAGIC %md ###Step 2b: Create Model Serving Endpoint
# MAGIC
# MAGIC With our model deployed to the MLFlow registry, we can now push the model to a Databricks model serving endpoint.  This endpoint will enable vector store population later in this notebook:

# COMMAND ----------

# DBTITLE 1,Configuration Values for Model Serving Endpoint
#name used to reference the model serving endpoint
endpoint_name = config['embedding_model_name']

# get url of this workspace where the model serving endpoint will be deployed
workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md To deploy our model serving endpoint, we may either use the model serving UI or the Databricks (administrative) REST API as documented [here](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html).  We have elected to use the REST API as it simplifies the deployment process.  However, it does require that we employ either a personal access token or a service principal in order to authenticate to that API.  We've elected to use a personal access token for the sake of simplicity but recommend the use of service principals in all production deployments.  More information on authentication options are available [here](https://docs.databricks.com/en/dev-tools/auth.html).
# MAGIC
# MAGIC The personal access token we have setup is secured as a Databricks secret with a scope of *llm_recommmender* and key of *embedding_model_endpoint_pat*.  More details on setting up such a secret is found [here](https://docs.databricks.com/en/security/secrets/index.html) but the basic Databricks CLI commands to set this up are as follows:
# MAGIC
# MAGIC ```
# MAGIC databricks secrets create-scope llm_recommender
# MAGIC databricks secrets put-secret llm_recommender embedding_model_endpoint_pat
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Authentication for Databricks REST API
# personal access token used by model serving endpoint to retrieve the model from mlflow registry
token = dbutils.secrets.get(scope="llm_recommender", key="embedding_model_endpoint_pat")

# COMMAND ----------

# MAGIC %md Using this information, we can now make the necessary calls required to deploy a model serving endpoint within Databricks.  Using the [serving-endpoint](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#api-workflow) endpoint, we will first check to see if the endpoint has already been deployed.  If it has not, we will deploy the endpoint and wait for it to enter into a ready state:
# MAGIC
# MAGIC **NOTE** In the code below, we are configuring the endpoint to stay awake, *i.e.* NOT scale to zero.  This will incur on-going charges.  Be sure to use the code at the bottom of this notebook to drop the endpoint once you are done working through this solution accelerator.

# COMMAND ----------

# DBTITLE 1,Deploy Model Serving Endpoint
# header for databricks rest api calls
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

# databricks rest api endpoint for deployment
base_url = f'{workspace_url}/api/2.0/serving-endpoints'

# does the model serving endpoint already exist?
results = requests.request(method='GET', headers=headers, url=f"{base_url}/{endpoint_name}")

# if endpoint exists ...
if results.status_code==200:
  print('endpoint already exists')

# otherwise, create an endpoint
else:
  print('creating endpoint')

  # configuration for model serving endpoint
  endpoint_config = {
    "name": endpoint_name,
    "config": {
      "served_models": [{
        "name": f"{config['embedding_model_name'].replace('.', '_')}_{1}",
        "model_name": config['embedding_model_name'],
        "model_version": model_version,
        "workload_type": "CPU",
        "workload_size": "Small",
        "scale_to_zero_enabled": False, # you may want to set this to false to minimize startup times
      }]
    }
  }

  # convert dictionary to json
  endpoint_json = json.dumps(endpoint_config, indent='  ')

  # send json payload to databricks rest api
  deploy_response = requests.request(method='POST', headers=headers, url=base_url, data=endpoint_json)

  # get response from databricks api
  if deploy_response.status_code != 200:
    raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')


# wait for endpoint to get into responsive state
timeout_seconds = 30 * 60 # minutes * seconds/minute 
stop_time = time.time() + timeout_seconds
waiting = True

while time.time() <= stop_time:

  # check on status of endpoint
  query_response = requests.request(method='GET', headers=headers, url=f"{base_url}/{endpoint_name}")

  status = query_response.json()['state']['ready']
  print(f"endpoint status: {status}")

    # if status is not ready, then sleep and try again
  if status=='NOT_READY':
    time.sleep(30)
  else: # otherwise stop looping
    waiting = False
    break

if waiting:
  raise Exception(f'Timeout expired waiting for endpoint to achieve a ready state.  Consider elevating the timeout setting.')

# COMMAND ----------

# MAGIC %md With the endpoint persisted, we can now verify it is producing embeddings for us:
# MAGIC
# MAGIC **NOTE** If you have previously setup the endpoint and configured it with *scale_to_zero_enabled* set to True, the endpoint may be asleep when you request a response below.  Setting an appropriate timeout will allow the endpoint time to wake up and respond.

# COMMAND ----------

# DBTITLE 1,Assemble Testing Payload for Endpoint
# assemble a few test sentences
sentences = ['This is a test', 'This is only a test']

# assemble them as a payload as expected by the model serving endpoint
ds_dict = {'dataframe_split': pd.DataFrame(pd.Series(sentences)).to_dict(orient='split')}
data_json = json.dumps(ds_dict, allow_nan=True)

print(data_json)

# COMMAND ----------

# DBTITLE 1,Test the Endpoint
invoke_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
invoke_url = f'{workspace_url}/serving-endpoints/{endpoint_name}/invocations'

# test the model serving endpoint with the assembled testing data
invoke_response = requests.request(method='POST', headers=invoke_headers, url=invoke_url, data=data_json, timeout=300)

# display results from endpoint
if invoke_response.status_code != 200:
  raise Exception(f'Request failed with status {invoke_response.status_code}, {invoke_response.text}')

print(invoke_response.text)

# COMMAND ----------

# MAGIC %md ###Step 2c: Populate Vector Store
# MAGIC
# MAGIC With our embedding model deployed to a model serving endpoint, we can now define a workflow to convert data in our products table into entries in our vector store.  The index creation and maintenance will take place as part of an on-going automation.  It will detect changes to our products table by reading the change log associated with it.  To ensure the [change log](https://docs.databricks.com/en/delta/delta-change-data-feed.html) is enabled, we can alter the table's definition as follows:
# MAGIC
# MAGIC **NOTE** Change detection only works with tables persisted in the Delta Lake format.

# COMMAND ----------

# DBTITLE 1,Enable the Change Log on our Product Table
_ = spark.sql("ALTER TABLE products SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md We can now configure a job to convert data into embeddings on an ongoing basis. This is done by creating a referencable endpoint for the vector store and and index associated with it:

# COMMAND ----------

# DBTITLE 1,Instantiate Vector Search Client
vs_client = VectorSearchClient()

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint
#name used for vector search endpoint
endpoint_name = 'vs_'+config['embedding_model_name']

# check if exists
endpoint_exists = True
try:
    vs_client.get_endpoint(endpoint_name)
except:
    pass
    endpoint_exists = False

# create vs endpoint
if not endpoint_exists:
    vs_client.create_endpoint(
        name=endpoint_name,
        endpoint_type="STANDARD" # or PERFORMANCE_OPTIMIZED, STORAGE_OPTIMIZED
    )

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
# check if index exists
index_exists = False
try:
  vs_client.get_index(index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}", endpoint_name=endpoint_name)
  index_exists = True
except:
  print('Ignoring error message ...')
  pass


if not index_exists:
  # connect delta lake table to vector store index table
  vs_client.create_delta_sync_index(
    endpoint_name=endpoint_name,
    source_table_name=f"{config['catalog']}.{config['schema']}.products",
    primary_key="id", # primary identifier in source table
    embedding_source_column="text", # field to index in source table
    index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}",
    pipeline_type='TRIGGERED',
    embedding_model_endpoint_name = config['embedding_model_name'] # model serving endpoint to use to create the embeddings
    )

# COMMAND ----------

# MAGIC %md The indexing works as part of a background job. It will take some time for the job to launch and start generating embeddings. While we are waiting for this, the job will be in a *provisioning* state.  We will need to wait for this to complete before proceeding with the remainder of this notebook:

# COMMAND ----------

# DBTITLE 1,Wait for Vector Store Index to Start Processing Data
timeout_seconds = 120 * 60  # minutes * seconds/minute
stop_time = time.time() + timeout_seconds
waiting = True

# get index
idx = vs_client.get_index(index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}", endpoint_name=endpoint_name)

# wait for index to complex indexing
while time.time() <= stop_time:

  # get state of index
  is_ready = idx.describe()['status']['ready']

  # if not ready, wait ...
  if is_ready:
    print('Ready')
    waiting = False
    break
  else:
    print('Waiting...')
    time.sleep(60)
   
# if exited loop because of time out, raise error
if waiting:
  raise Exception(f'Timeout expired waiting for index to be provisioned.  Consider elevating the timeout setting.')

# COMMAND ----------

# MAGIC %md ##Step 3: Search the Vector Store
# MAGIC
# MAGIC With the vector store populated, we might perform a simple search as follows:

# COMMAND ----------

# DBTITLE 1,Locate Relevant Content in Vector Store Index
# connect to index
idx = vs_client.get_index(index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}", endpoint_name=endpoint_name)

# search the vector store for related items
idx.similarity_search(
  query_text = "winter boots",
  columns = ["id", "text"], # columns to return
  num_results = 5
  )

# COMMAND ----------

# MAGIC %md You'll notice the basic search results include a lot of metadata.  If you want to get just the retrieved items, this info is found under the `results` and `data array` keys:

# COMMAND ----------

# DBTITLE 1,Get Just the Results
search_results = idx.similarity_search(
  query_text = "winter boots",
  columns = ["id", "text"], # columns to return
  num_results = 5
  )

print(search_results['result']['data_array'])

# COMMAND ----------

# MAGIC %md Please notice that the vector search doesn't necessarily match based on word matches but instead maps the provided query text into a embedding that represents how the provided item relates to concepts it's learned by processing large volumes of text.  As a result, it's possible that the search might find *related* items that don't exactly match the search term but which have a more general association.  For this type of recommender, this kind of expansive search is fine as we are attempting to not match exact items but instead leverage these loose associations to expand the set of relevant products we might put in front of a customer.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |