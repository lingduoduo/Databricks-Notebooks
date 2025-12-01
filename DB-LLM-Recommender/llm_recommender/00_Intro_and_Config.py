# Databricks notebook source
# MAGIC %md The purpose of this notebook is to introduce the LLM Recommender solution accelerator and provide configuration settings for the various notebooks that comprise it.  This notebook was developed using a Databricks ML 14.2 cluster. 

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC Recommenders employ various strategies to suggest products that may appeal to a customer. With this recommender, we are considering a set of products a customer has already purchased or otherwise shown a preference for and are using our knowledge of those products to suggest additional, relevant items.  
# MAGIC
# MAGIC The association between these two sets of products, *i.e.* those that are preferred and those that are suggested, is based on a general understanding of how items relate within a given culture.  For example, for many consumers if we understood they were purchasing a pair of *gloves*, a *scarf* and a *coat*, this might suggest *winter boots*, an *insulated hat* or *ear muffs* might be of interest as these items conceptually relate to providing protection from the cold.
# MAGIC
# MAGIC This type of recommender does not take into account relationships between purchased items as might be observed within an organization’s historical data, so that the famous (though spurious) [*diapers and beer*](https://tdwi.org/articles/2016/11/15/beer-and-diapers-impossible-correlation.aspx) pattern might never emerge.  Still, this recommender reflects how one reasonable individual might suggest items to another and may provide a helpful and personable experience within some contexts.
# MAGIC

# COMMAND ----------

# MAGIC %md The other limitation of this approach is that the items being suggested are highly generalized, *e.g.* *winter boots* is a broad category of products and not a specific SKU.  In addition, the LLM has no knowledge of whether a suggested item even is available for purchase through a given retail outlet.
# MAGIC
# MAGIC To overcome this limitation, we need to intersect the generalized suggested items with the specific items in our product inventory.  A general recommendations of *winter boots* might align nicely with various *snow boots*, *insulated boots* or even *waterproof galoshes* found within in our product catalog depending on our tolerance around item similarity.
# MAGIC
# MAGIC To support this, we can take the descriptive information associated with each of the products in our inventory and convert them (using an LLM) into an embedding.  The embedding captures a mapping of each item to the various *concepts* learned by the LLM as it was exposed to training data.  We can then convert a generalized suggestion such as *winter boots* into an embedding using the same model and calculate the similarity between the suggested product embedding and the embeddings of various items in our product inventory to identify those items most closely *conceptually* related to that suggestion.
# MAGIC

# COMMAND ----------

# MAGIC %md With these concepts in mind, we will tackle this solution accelerator by first connecting to an LLM and developing a prompt to trigger generalized product suggestions.  We will then convert our product details into a searchable database of embeddings.  And then finally, we will bring these two halves of the solution together to enable the deployment of a robust recommender into an application infrastructure.

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC
# MAGIC The following settings are used across the various notebooks found in this solution accelerator:

# COMMAND ----------

# DBTITLE 1,Instantiate Config
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Product Database
# set catalog
config['catalog'] = 'solacc_uc'
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {config['catalog']}")
_ = spark.sql(f"USE CATALOG {config['catalog']}")

# set schema
config['schema'] = 'llm_recommender'
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config['schema']}")
_ = spark.sql(f"USE SCHEMA {config['schema']}")

# COMMAND ----------

# DBTITLE 1,Vector Search Index
config['vs index'] = 'product_index'

# COMMAND ----------

# DBTITLE 1,Model
config['embedding_model_name'] = 'llm_recommender_embeddings'

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |