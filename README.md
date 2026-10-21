**This code was run in Google CoLab using the PyTorch library.**

# What is this project for?

The FER2013 Facial Recognition project was designed for caretakers/paraprofessionals of neurodivergent kids. Neurodivergent is a broad term referring to individuals with mental disorders, special needs, and other developmental or behavioral conditions. Specifically, this project was created to address alexithymia, or “emotional blindness,” in those particular individuals, as they cannot properly express or communicate their emotions. In this case, caretakers can use the technology present in this project to help those individuals express those emotions.

Specifically, the goal of this project is to answer an overarching question along with three other questions: How can we use a model trained from scratch to help neurodivergent people express their emotions using facial recognition technology? Can deep-learning be used as a solution? Will neurodivergent people have a better chance at expressing their emotions? Can those emotions be handled better? 

Ultimately, there is a great potential with further improvement for this model to be deployed in the real world; the accuracy is not excellently high at 68.18%, but the goal was 60-70% originally. The model fares decently well with other pretrained models, with EfficientNet scoring the highest test accuracy. Realistically, the model can confuse certain emotions with each other, such as the pairs of anger and disgust or fear and surprise. The lack of depth in the training will not necessarily prepare the model for handling more complex emotions, as it was trained and tested with only 7 classes. 

However, in the current stage of AI and machine learning, it is crucial to note that this technology can be potentially dangerous in the case of mass surveillance. Those high in power who gain access to advanced facial recognition have the ability to put civilians’ privacy and security at risk; therefore, the use of this potential model needs to be limited and regulated.

More information can be found [here.](https://github.com/jt187276/FER2013-Facial-Recognition/blob/main/Facial%20Recognition.pptx)

# Tutorial
We ran this entire experiment in a Google Colab environment. To replicate our results, follow these instructions by running each Python Notebook in Google Colab.

1. Copy boilerplate/fer_common.py and all the Python Notebooks in the workflow directory into a root directory for your Google Drive.

2. fer_common.py contains much of the boilerplate training/testing code that our notebooks refer to. For example, BackboneClassifier is a wrapper class around a specific model instance that adapts a general model to our use case (e.g. transforming 1000 class outputs to 7 classes, removing auxiliary branches from models like inception_v3, etc.)

3. Run `workflow/01_prepare_data.ipynb` in a Google Colab environment. This will fetch the FER2013 dataset, perform an 80/20 stratified split on the training data to produce a training and validation dataset, and will save the split training, validation, and testing datasets (FER2013 provides a test dataset already) to your root directory in Google Drive.

4. For each of the 9 models of Cell 8 in `workflow/02_train_model.ipynb`, set MODEL_NAME to the name of the model you are currently training (see cell comments for exact model names). Run the entire notebook to train the model.

5. This notebook will import the default, pretrained weights associated with the model, and fine-tune it with the FER2013 dataset for transfer learning to our use case.

6. We structure the notebook this way to split up the model training portion of this experiment into individual sessions to not lose progress for convenience. Once a model is trained, its weights and training history will be saved to the boilerplate directory. 

7. After training a model, we recommend using the existing Colab runtime after training and changing the model name in cell 8, and running all cells underneath it to train a different model. This avoids the overhead of having to import the FER2013 split from drive per session.

8. After all models are trained, run `workflow/03_evaluate.ipynb` to evaluate all models against FER2013’s testing dataset. This will produce per-model accuracy, F1, precision, and recall scores, as well as visualizations of their performance and training history.
