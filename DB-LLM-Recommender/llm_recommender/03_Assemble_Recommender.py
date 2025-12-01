# Databricks notebook source
# MAGIC %md The purpose of this notebook is to assemble the LLM-based product recommender for intergration into various applications.  This notebook was developed on a Databricks ML 14.2 cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC With our product catalog indexed for search and a general recommender in place, we now have all the components we need to enable our recommendation engine. The basic pattern will see us receiving a list of items from an external application, generating a set of general recommendations from these items and then using those recommendations to bring forward the specific items in our catalog that we can then present to a user.  All the basic components for this were developed in prior notebooks.  Our primary task here is to bring these together behind a single function call.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-vectorsearch databricks-genai-inference mlflow[genai]>=2.9.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from databricks_genai_inference import ChatCompletion
from databricks.vector_search.client import VectorSearchClient

import mlflow
import mlflow.deployments

import pandas as pd
import json
import os
import requests

# COMMAND ----------

# MAGIC %md ##Step 1: Combine General Recommendation & Product Search
# MAGIC
# MAGIC As mentioned above, all the logic for generating general recommendations and retrieving specific, related products from our product catalog were defined in prior notebooks.  Here, we will refactor that logic into a series of functions and make an end-to-end call to retrieve recommendations.  We will also include references to LLM model endpoints and workspace URLs in anticipation of a deployment from outside of the Databricks environment in the next step:
# MAGIC
# MAGIC **NOTE** The url for your LLM is accessible through the model serving page of your Databricks workspace assuming this feature is enabled in your region.
# MAGIC
# MAGIC **NOTE** Connectivity to your vector store and LLM endpoint from outside of Databricks depends on either a [personal access token](https://docs.databricks.com/en/dev-tools/auth/pat.html) or a [service principle](https://docs.databricks.com/en/dev-tools/service-principals.html).  We've elected to use personal access token (PAT) and have stored it to a [Databricks secrets](https://docs.databricks.com/en/security/secrets/index.html).  The secret scope and key are identified in the variables below.

# COMMAND ----------

# DBTITLE 1,Define Workspace Parameters
workspace_url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}' # the url of the workspace housing the vector store
endpoint_url = f'{workspace_url}/serving-endpoints/databricks-llama-2-70b-chat/invocations' # the url associated with your llm
token = dbutils.secrets.get('llm_recommender','vs_pat')

# COMMAND ----------

# DBTITLE 1,Define Items from which to Make Recommendations
ordered_list_of_items = ['scarf', 'beanie', 'ear muffs']

# COMMAND ----------

# DBTITLE 1,Get General Product Suggestions
def get_general_products(endpoint_url, token, items):

  # define function to assemble the prompt
  def _get_prompt(items):

    # define system prompt
    system_prompt = 'You are an AI assistant functioning as a recommendation system for an ecommerce website.'
    system_prompt += ' Be specific and limit your answers to the requested format.'

    # build user prompt    
    items_delimited = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
    user_prompt = f"A user bought {items_delimited} in that order. What five items would he/she be likely to purchase next?"
    user_prompt += "Express your response as a JSON object with a key of 'next_items' and a value representing your array of recommended items."
    
    # assemble full prompt
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
    prompt += f"{user_prompt}[/INST]"
    
    return prompt


  # headers for the request
  headers = {
      'Content-Type': 'application/json',
   }

  # prepare the data following the structure provided in the example
  prompt = _get_prompt(items)
  data = {
      "messages": [
          {
              "role": "user",
              "content": prompt
          }
      ],
      "max_tokens": 128
  }

  # get the authentication token from environment variables
  # replace the following with your actual token or retrieve from an environment variable
  auth_token = token

  # make the POST request
  response = requests.post(endpoint_url, headers=headers, json=data, auth=('token', auth_token))
  raw_text = json.loads(response.text)['choices'][0]['message']["content"]

  # extract just the list from the response
  json_text = raw_text[:raw_text.rindex('}')+1]
  ret = json.loads(json_text)['next_items']

  # return the response
  return ret



# test the function
get_general_products(endpoint_url, token, ordered_list_of_items)

# COMMAND ----------

# DBTITLE 1,Get Specific Product Suggestions
def get_specific_products(vs_client, general_items, num_items):

  # connect to vector store index
  idx = vs_client.get_index(
  	index_name=f"{config['catalog']}.{config['schema']}.{config['vs index']}", 
  	endpoint_name='vs_'+config['embedding_model_name']
  	)

  # retrieve related products
  results = idx.similarity_search(
	  query_text=str(general_items), # pass all items in as a string
	  columns=["id", "text"],
	  num_results=num_items
	  )
	  
   # return search results
  return [item[-2] for item in results['result']['data_array']]
 
# connect to vector store
vs_client = VectorSearchClient(personal_access_token=token, workspace_url=workspace_url)

# get general item recommendations
general_items = get_general_products(endpoint_url, token, ordered_list_of_items)

# get specific items
get_specific_products(vs_client, general_items, 5)

# COMMAND ----------

# MAGIC %md ##Step 2: Deploy the Model
# MAGIC
# MAGIC We now have all the building blocks we need to deploy our recommender.  We will deploy that model using [Fast API](https://fastapi.tiangolo.com/), a simple and lightweight mechanism for deploying Python-based APIs.  We've elected to use this for our deployment path as many developers are using Fast API for such deployments and the model serving capabilities in Databricks for generative AI models are evolving.  In the future, we anticipate having a preference for model serving, but we figured this would be a great opportunity to demonstrate deployment using Fast API.

# COMMAND ----------

# MAGIC %md The core function behind our Fast API deployment will be 
# MAGIC
# MAGIC ```
# MAGIC app = FastAPI()
# MAGIC
# MAGIC # define the request body model
# MAGIC class ItemList(BaseModel):
# MAGIC     items: List[str]
# MAGIC
# MAGIC # FastAPI endpoint to handle POST requests for item recommendations
# MAGIC @app.post("/recommendations/")
# MAGIC async def get_recommendations(items: ItemList):
# MAGIC     vs_client = VectorSearchClient(personal_access_token=token, workspace_url=workspace_url)
# MAGIC     n_items=5
# MAGIC     try:
# MAGIC         result = LLM4rec(vs_client, endpoint_url, token, items.items, n_items)
# MAGIC         return result
# MAGIC     except Exception as e:
# MAGIC         raise HTTPException(status_code=500, detail=str(e))
# MAGIC ```
# MAGIC
# MAGIC You can see from this function call that we will need to bring our code together behind a function we will call LLM4rec.  This function will accept an ItemList, a user-defined class that's nothing more than a list of strings, along with a count of items to return.  
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC We will assemble this function an all its dependent logic within a file we will call *app.py*.  The complete contents of *app.py* are as follows:
# MAGIC
# MAGIC ```
# MAGIC from fastapi import FastAPI, HTTPException
# MAGIC from pydantic import BaseModel
# MAGIC import requests
# MAGIC import pandas as pd
# MAGIC import json
# MAGIC import random
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from typing import List, Dict 
# MAGIC import os
# MAGIC
# MAGIC # CONFIGURATIONS
# MAGIC # ----------------------------------------------
# MAGIC workspace_url = os.environ['LLM_RECOMMENDER_WORKSPACE_URL'] # the url of the workspace housing the vector store
# MAGIC endpoint_url = os.environ['LLM_RECOMMENDER_ENDPOINT_URL'] # the url associated with your llm
# MAGIC token = os.environ['LLM_RECOMMENDER_PAT']
# MAGIC # ----------------------------------------------
# MAGIC
# MAGIC # GENERAL PRODUCT RECOMMENDATIONS
# MAGIC # ----------------------------------------------
# MAGIC def get_general_products(endpoint_url, token, items):
# MAGIC
# MAGIC   # define function to assemble the prompt
# MAGIC   def _get_prompt(items):
# MAGIC
# MAGIC     # define system prompt
# MAGIC     system_prompt = 'You are an AI assistant functioning as a recommendation system for an ecommerce website.'
# MAGIC     system_prompt += ' Be specific and limit your answers to the requested format.'
# MAGIC
# MAGIC     # build user prompt    
# MAGIC     items_delimited = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
# MAGIC     user_prompt = f"A user bought {items_delimited} in that order. What five items would he/she be likely to purchase next?"
# MAGIC     user_prompt += "Express your response as a JSON object with a key of 'next_items' and a value representing your array of recommended items."
# MAGIC     
# MAGIC     # assemble full prompt
# MAGIC     prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
# MAGIC     prompt += f"{user_prompt}[/INST]"
# MAGIC     
# MAGIC     return prompt
# MAGIC
# MAGIC
# MAGIC   # headers for the request
# MAGIC   headers = {
# MAGIC       'Content-Type': 'application/json',
# MAGIC    }
# MAGIC
# MAGIC   # prepare the data following the structure provided in the example
# MAGIC   prompt = _get_prompt(items)
# MAGIC   data = {
# MAGIC       "messages": [
# MAGIC           {
# MAGIC               "role": "user",
# MAGIC               "content": prompt
# MAGIC           }
# MAGIC       ],
# MAGIC       "max_tokens": 128
# MAGIC   }
# MAGIC
# MAGIC   # get the authentication token from environment variables
# MAGIC   # replace the following with your actual token or retrieve from an environment variable
# MAGIC   auth_token = token
# MAGIC
# MAGIC   # make the POST request
# MAGIC   response = requests.post(endpoint_url, headers=headers, json=data, auth=('token', auth_token))
# MAGIC   raw_text = json.loads(response.text)['choices'][0]['message']["content"]
# MAGIC
# MAGIC   # extract just the list from the response
# MAGIC   json_text = raw_text[:raw_text.rindex('}')+1]
# MAGIC   ret = json.loads(json_text)['next_items']
# MAGIC
# MAGIC   # return the response
# MAGIC   return ret
# MAGIC # ----------------------------------------------
# MAGIC
# MAGIC # SPECIFIC PRODUCT RECOMMENDATIONS
# MAGIC # ----------------------------------------------
# MAGIC def get_specific_products(vs_client, general_items, num_items):
# MAGIC
# MAGIC   # connect to vector store index
# MAGIC   idx = vs_client.get_index(
# MAGIC   	index_name='solacc_uc.llm_recommender.product_index',  # replaced config values here 
# MAGIC   	endpoint_name='vs_llm_recommender_embeddings' # replaced config values here
# MAGIC   	)
# MAGIC
# MAGIC   # retrieve related products
# MAGIC   results = idx.similarity_search(
# MAGIC 	  query_text=str(general_items), # pass all items in as a string
# MAGIC 	  columns=["id", "text"],
# MAGIC 	  num_results=num_items
# MAGIC 	  )
# MAGIC 	  
# MAGIC    # return search results
# MAGIC   return [item[-2] for item in results['result']['data_array']]
# MAGIC # ----------------------------------------------
# MAGIC
# MAGIC # LL4REC
# MAGIC # ----------------------------------------------
# MAGIC def LLM4rec(vs_client, endpoint_url, token, items, n_items):
# MAGIC
# MAGIC   # get general item recommendations
# MAGIC   general_items = get_general_products(endpoint_url, token, items)
# MAGIC
# MAGIC   # get specific items
# MAGIC   specific_items = get_specific_products(vs_client, general_items, n_items)
# MAGIC
# MAGIC   return specific_items
# MAGIC # ----------------------------------------------
# MAGIC
# MAGIC
# MAGIC app = FastAPI()
# MAGIC
# MAGIC # define the request body model
# MAGIC class ItemList(BaseModel):
# MAGIC     items: List[str]
# MAGIC
# MAGIC # FastAPI endpoint to handle POST requests for item recommendations
# MAGIC @app.post("/recommendations/")
# MAGIC async def get_recommendations(items: ItemList):
# MAGIC     n_items=5
# MAGIC     vs_client = VectorSearchClient(personal_access_token=token, workspace_url=workspace_url)
# MAGIC     try:
# MAGIC         result = LLM4rec(vs_client, endpoint_url, token, items.items, n_items)
# MAGIC         return result
# MAGIC     except Exception as e:
# MAGIC         raise HTTPException(status_code=500, detail=str(e))
# MAGIC ```

# COMMAND ----------

# MAGIC %md **NOTE** Please note that the workspace URL, LLM endpoint URL and the personal access tokens in the *app.py* file are now accessed through environment variables that you will need to set within your host environment.
# MAGIC
# MAGIC **NOTE** In the earlier code examples, we used a number of values retrieved from a *config* dictionary.  You will need to either hardcode these values or map these to environment variables in *app.py*.

# COMMAND ----------

# MAGIC %md The *app.py* file will be deployed to a folder on the host from which the API will be served. A *requirements.txt* will reside in this folder to define Python dependencies.  The contents of this file are as follows:
# MAGIC
# MAGIC ```
# MAGIC annotated-types==0.6.0
# MAGIC anyio==4.3.0
# MAGIC certifi==2024.2.2
# MAGIC charset-normalizer==3.3.2
# MAGIC click==8.1.7
# MAGIC cloudpickle==3.0.0
# MAGIC databricks-cli==0.18.0
# MAGIC databricks-vectorsearch==0.22
# MAGIC entrypoints==0.4
# MAGIC exceptiongroup==1.2.0
# MAGIC fastapi==0.109.2
# MAGIC gitdb==4.0.11
# MAGIC GitPython==3.1.42
# MAGIC h11==0.14.0
# MAGIC idna==3.6
# MAGIC importlib-metadata==7.0.1
# MAGIC mlflow-skinny==2.10.2
# MAGIC numpy==1.26.4
# MAGIC oauthlib==3.2.2
# MAGIC packaging==23.2
# MAGIC pandas==2.2.0
# MAGIC protobuf==4.25.3
# MAGIC pyarrow==15.0.0
# MAGIC pydantic==2.6.1
# MAGIC pydantic_core==2.16.2
# MAGIC PyJWT==2.8.0
# MAGIC python-dateutil==2.8.2
# MAGIC pytz==2023.4
# MAGIC PyYAML==6.0.1
# MAGIC requests==2.31.0
# MAGIC six==1.16.0
# MAGIC smmap==5.0.1
# MAGIC sniffio==1.3.0
# MAGIC sqlparse==0.4.4
# MAGIC starlette==0.36.3
# MAGIC tabulate==0.9.0
# MAGIC typing_extensions==4.9.0
# MAGIC tzdata==2024.1
# MAGIC urllib3==2.2.1
# MAGIC uvicorn==0.27.1
# MAGIC zipp==3.17.0
# MAGIC ```

# COMMAND ----------

# MAGIC %md To setup a testing environment, execute the following:
# MAGIC
# MAGIC **NOTE** We used a Ubuntu 22.04.2 LTS environment for testing on our end
# MAGIC
# MAGIC ```
# MAGIC # move to folder where you wish to create your virtual environment
# MAGIC # copy the app.py and requirements.txt files here
# MAGIC # set your environment variables
# MAGIC
# MAGIC python3 -m venv venv
# MAGIC source venv/bin/activate
# MAGIC pip install -r requirements.txt
# MAGIC uvicorn app:app --reload
# MAGIC ```
# MAGIC
# MAGIC If successful, you should be greated with some output that looks like the following:
# MAGIC ```
# MAGIC INFO:     Will watch for changes in these directories: ['/home/.../...']
# MAGIC INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# MAGIC INFO:     Started reloader process [774] using StatReload
# MAGIC INFO:     Started server process [776]
# MAGIC INFO:     Waiting for application startup.
# MAGIC INFO:     Application startup complete.
# MAGIC INFO:     127.0.0.1:50170 - "GET /docs HTTP/1.1" 200 OK
# MAGIC INFO:     127.0.0.1:50170 - "GET /openapi.json HTTP/1.1" 200 OK
# MAGIC ```

# COMMAND ----------

# MAGIC %md With the application running, you can now open a browser on the local machine to http://127.0.0.1:8000/docs.  There, you should be greated with a Swagger UI that accepts item inputs and returns recommendations:
# MAGIC
# MAGIC **NOTE** Click the *Try It Out* button to enable inputs and testing.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/llmrec_fastapi_swagger.png'>

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |