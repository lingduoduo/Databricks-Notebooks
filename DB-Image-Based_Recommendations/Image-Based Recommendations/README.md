# How to Build an Image Recommendation System for Online Retail using Contrastive Learning (and at scale!)

In this solution, you will learn the end to end process of building a recommender that uses a model trained using similarity learning, a novel machine learning approach more suitable for finding similar items. You will use the Tensorflow_similarity library to train the model and Spark,Horovod, and Hypeopt, to scale the model training across a GPU cluster. Mlflow will be used to log and track all aspects of the process and Delta will be used to preserve data lineage and reproducibility.

At a high level, similarity models are trained using contrastive learning. In contrastive learning, the goal is to make the machine learning model (an adaptive algorithm) learn an embedding space where the distance between similar items is minimized and distance between dissimilar items is maximized. In this quickstart we will use the fashion MNIST dataset, which comprises of around 70,000 images of various clothing items. Based on the above description, a similarity model trained on this labelled dataset will learn an embedding space where embeddings of similar items e.g. boots are closer together and different items e.g. boots and bandanas are far apart. 

In similarity learning, the goal is to teach the model to discover a space where the similar items are grouped closer to each other and dissimilar items are separated even more. In supervised similarity learning, the algorithms has access to image labels to learn from, in addition to the image data itself.
___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library / data source                  | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| tensorflow                                | package                 | Apache 2.0  | https://github.com/tensorflow/tensorflow/blob/master/LICENSE  |
| fashion-mnist| dataset | MIT | https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE |

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
