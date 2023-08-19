# Sentiment Analysis on Yelp reviews using Deep Learning Methods

This repository contains a comprehensive notebook that delves into the fundamental aspects of Natural Language Processing (NLP) for sentiment analysis. The notebook guides you through the entire process of analyzing Yelp reviews sentiment using various deep learning techniques. The breakdown of the tasks covered is as follows:

## Key Sections of the Notebook:

1. **Data Processing**: The raw data is processed by converting it into a pandas dataframe. It is then manipulated and stored in a CSV file according to specific requirements.

2. **Data Vectorization**: Text reviews are transformed into integer vectors using one-hot encoding. This is essential since deep learning models only accept numerical inputs.

3. **Data Vocabulary**: The creation of a vocabulary is crucial for NLP tasks. The notebook explains how a vocabulary is constructed to keep track of word frequencies and positions in the text.

4. **Data Processing**: PyTorch's DataLoader handles the heavy lifting of data processing by batching and shuffling data, allowing you to focus on the model. It efficiently loads data in parallel using multiprocessing workers.

5. **Deep Learning Models**: Two types of models are explored:
   - Single Layer Perceptron:
      - Utilizes a linear layer with a softmax activation
      - Uses the sigmoid activation function
      - Optimized with the Adam optimizer
      - Binary cross-entropy loss for binary output
   - Multi-Layer Perceptron:
      - Comprises two linear layers
      - Employs sigmoid activation
      - Optimized with Adam optimizer
      - Binary cross-entropy loss
   - Convolutional Neural Network
     - Using 1D convolutions for sequence-based analysis.

6. **Training, Validation, and Testing Loop**: The notebook outlines the process of training, validating, and testing the models, providing insights into their performance.

Includes **Hyperparameters Tuning and Understanding**: Understanding the importance of hyperparameters and their impact on model performance is discussed in depth.

Include Implementation of CUDA in PyTorch to leverage GPU acceleration.

## How to Run:

1. Clone the repository.
2. Download the yelp review dataset from kaggle(https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset) and store it in the data/yelp/raw_test.csv and data/yelp/raw_train.csv locations.(or change the location address in code)
3. Run the following commands:
   ```
   cd yelp_reviews
   jupyter notebook
   ```

## Insights and Observations:

During the analysis, the following observations were made:
- The simple perceptron performs well on a lightweight dataset.
- When using the complete Yelp reviews dataset, the perceptron overfits, yielding an accuracy of 100%.

## Ongoing Improvements:

The notebook continues to evolve with planned enhancements including:
- Inclusion of code for saving vectorized data and model files.
- Further exploration of model predictions.
- Implementation of additional deep learning models.

Feel free to explore and enhance the notebook to deepen your understanding of sentiment analysis and its applications in NLP!