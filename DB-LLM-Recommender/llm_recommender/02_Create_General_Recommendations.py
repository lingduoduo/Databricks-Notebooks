# Databricks notebook source
# MAGIC %md The purpose of this notebook is to connect to an LLM in order to create generalized product recommendations.  This notebook was developed on a Databricks ML 14.2 cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC The next step in building our recommender is to tap into the power of a general purpose large language model (LLM) to suggest additional items for a user.  Given knowledge of the items a customer has purchased, browsed or otherwise expressed interest in, the LLM can tap a resevoir of knowledge to suggest what items would typically be associated with these.
# MAGIC
# MAGIC The steps involved with getting this part of the application stood app are simply to connect to an appropriate LLM and tune a prompt that returns the right results in the right format.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-genai-inference
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import the Required Libraries
from databricks_genai_inference import ChatCompletion

# COMMAND ----------

# MAGIC %md ##Step 1: Connect to the LLM
# MAGIC
# MAGIC With the availability of a wide variety of proprietary services and open source models, we have numerous options for how we will address our LLM needs. Many frequently used models have been pre-provisioned for use within Databricks as part of our [foundation model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html).  This includes [Meta's Llama2-70B-Chat model](https://ai.meta.com/llama/) which is being widely adopted as a robust, general-purpose chat enabler.
# MAGIC
# MAGIC Because all the plumbing has already been put in place, connectivity to this model is very simple.  We just make the call:

# COMMAND ----------

# DBTITLE 1,Text Connectivity to the LLM
response = ChatCompletion.create(
  model='llama-2-70b-chat',
  messages=[
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Knock knock.'}
    ],
  max_tokens=128
  )

print(f'response.message:{response.message}')

# COMMAND ----------

# MAGIC %md ##Step 2:  Engineer the prompt
# MAGIC
# MAGIC The foundation model API greatly simplifies the process of not only connecting to a model but performing a chat task. Instead of constructing a prompt with specialized formatting, we can assemble a fairly standard [chat message payload](https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-task) to generate a response.
# MAGIC
# MAGIC As is typical in most chat applications, we will supply a system prompt and a user prompt.  The system prompt for our app might look like this:

# COMMAND ----------

# DBTITLE 1,Define System Prompt
system_prompt = 'You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format. Keep your answers short and concise.'

# COMMAND ----------

# MAGIC %md For the user prompt, we need to incorporate a list of items from which product recommendations will be produced.  Because this prompt is dynamic, it might be best to define the prompt using a function:

# COMMAND ----------

# DBTITLE 1,Define Function to Build User Prompt
# define function to create prompt produce a recommended set of products
def get_user_prompt(ordered_list_of_items):

   # assemble user prompt
  prompt = None
  if len(ordered_list_of_items) > 0:
    items = ', '.join(ordered_list_of_items)
    prompt =  f"A user bought the following items: {items}. What next ten items would he/she be likely to purchase next?"
    prompt += " Express your response as a JSON object with a key of 'next_items' and a value representing your array of recommended items."
 
  return prompt

# COMMAND ----------

# DBTITLE 1,Retrieve User Prompt
# get prompt and results
user_prompt = get_user_prompt(
    ['scarf', 'beanie', 'ear muffs', 'thermal underwear']
    )

print(user_prompt)

# COMMAND ----------

# MAGIC %md And now we can test the prompt with a call to our LLM:

# COMMAND ----------

# DBTITLE 1,Test the Prompt
response = ChatCompletion.create(
  model='llama-2-70b-chat',
  messages=[
    {'role': 'system', 'content': system_prompt},
    {'role': 'user','content': user_prompt}
    ],
  max_tokens=128
  )
print(f'response.message:{response.message}')

# COMMAND ----------

# MAGIC %md Getting the model to respond with lists in the structure  you want and with relevant content is tricky.  (For example, we couldn't get a valid dictionary structure by requesting a python dictionary but found that requesting a JSON object did the trick.) You'll need to experiment with a variety of wordings and phrasings to trigger the right results.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |