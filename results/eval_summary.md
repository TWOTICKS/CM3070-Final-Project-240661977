# RAG Prototype Evaluation Summary

Questions evaluated: 11 (10 in-scope, 1 out-of-scope)

## Retrieval quality (mean recall@3 / precision@3 by topic, in-scope questions)

- With cross-encoder re-ranking:    recall=1.00, precision=0.80
- Without re-ranking (embeds only): recall=1.00, precision=0.80

## Answer faithfulness (NLI entailment rate vs retrieved context)

Two scoring methods are compared:
- **concatenated**: all retrieved chunks joined into one premise (original method).
- **per-chunk**: each answer sentence is checked against each retrieved chunk
  separately, keeping the chunk with the highest entailment probability.

- RAG answers (grounded in retrieved context):
  - concatenated: mean faithfulness = 0.10, contradiction rate = 0.00
  - per-chunk:    mean faithfulness = 0.33, contradiction rate = 0.08
- No-retrieval baseline answers, checked against the SAME retrieved context:
  - concatenated: mean faithfulness = 0.03
  - per-chunk:    mean faithfulness = 0.18

## Answer-aware confidence gate

Post-generation gate (`RAGPipeline.low_confidence`): flags an answer when per-chunk faithfulness < 0.15 or per-chunk contradiction rate >= 0.5. This is separate from retrieval-based abstention and catches the case where retrieval looked confident enough to proceed but the generated answer still ended up poorly grounded.

- In-scope answers flagged low-confidence: 4 / 10
  - q3: faithfulness(per-chunk)=0.00, contradiction(per-chunk)=0.00
  - q5: faithfulness(per-chunk)=0.00, contradiction(per-chunk)=0.50
  - q8: faithfulness(per-chunk)=0.00, contradiction(per-chunk)=0.00
  - q10: faithfulness(per-chunk)=0.00, contradiction(per-chunk)=0.00

## Out-of-scope question handling

- Q: What did the lecturer say about quantum computing and its impact on machine learning?
  - top-3 rerank scores: [0.71, -7.34, -9.66]
  - abstained: False
  - RAG answer: The lecture excerpt does not mention anything about quantum computing and its impact on machine learning.
  - Baseline answer: The lecturer emphasized that quantum computing has the potential to revolutionize machine learning by potentially solving complex problems much faster than classical computers, especially in areas like optimization and cryptography. They also noted the ongoing research challenges and the need for interdisciplinary collaboration between computer scientists, physicists, and data scientists to fully realize this technology's benefits.

## Per-question detail (in-scope)

### q1: What is the difference between supervised, unsupervised, and reinforcement learning?
- relevant topic(s): ['Supervised, Unsupervised, and Reinforcement Learning']
- retrieved (reranked): ['Supervised, Unsupervised, and Reinforcement Learning', 'Supervised, Unsupervised, and Reinforcement Learning', 'Supervised, Unsupervised, and Reinforcement Learning']  (recall=1.00, precision=1.00)
- retrieved (no rerank): ['Supervised, Unsupervised, and Reinforcement Learning', 'Supervised, Unsupervised, and Reinforcement Learning', 'Supervised, Unsupervised, and Reinforcement Learning']  (recall=1.00, precision=1.00)
- RAG answer: Supervised learning involves training models with labeled data, aiming to map inputs to correct outputs. Unsupervised learning finds patterns in unlabeled data, grouping similar items together. Reinforcement learning teaches agents to maximize rewards through interaction with environments, optimizing strategies over repeated trials.
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.33, contradiction_rate=0.00
- Baseline answer: Supervised learning involves training models with labeled data to predict outcomes; Unsupervised learning finds patterns in unlabeled data without explicit guidance; Reinforcement learning teaches agents to make decisions by receiving rewards or penalties from their actions.
- Expected answer: Supervised learning trains on labelled examples (input plus correct output), unsupervised learning finds structure in unlabelled data (e.g. clustering), and reinforcement learning has an agent that learns a policy by taking actions in an environment and receiving rewards or penalties.

### q2: What is the ReLU activation function and why is it popular in deep learning?
- relevant topic(s): ['Neural Network Fundamentals']
- retrieved (reranked): ['Neural Network Fundamentals', 'Neural Network Fundamentals', 'Training Neural Networks - Gradient Descent and Backpropagation']  (recall=1.00, precision=0.67)
- retrieved (no rerank): ['Neural Network Fundamentals', 'Neural Network Fundamentals', 'Convolutional Neural Networks for Image Processing']  (recall=1.00, precision=0.67)
- RAG answer: The ReLU (Rectified Linear Unit) activation function is used in neural networks to introduce non-linearity. It outputs zero for any negative input and passes the absolute value of the input through for positive inputs. This makes it computationally efficient and helps prevent vanishing gradient problems common in deep networks with sigmoid activations. Its popularity stems from its simplicity, computational efficiency, and effectiveness in avoiding issues like exploding gradients.
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.25, contradiction_rate=0.00
- Baseline answer: The Rectified Linear Unit (ReLU) is an activation function commonly used in neural networks, especially in deep learning models. It outputs zero for negative inputs and the input value itself for positive inputs, which helps to introduce non-linearity into the model while simplifying computation. Its popularity stems from its simplicity, efficiency, and effectiveness in training deep networks, as well as its ability to mitigate the vanishing gradient problem seen with other activation functions like sigmoid or tanh.
- Expected answer: ReLU (rectified linear unit) outputs zero for negative inputs and passes positive inputs through unchanged. It is popular because it is computationally cheap and helps avoid training problems that sigmoid can cause in deep networks.

### q3: How does backpropagation work and why is it important for training neural networks?
- relevant topic(s): ['Training Neural Networks - Gradient Descent and Backpropagation']
- retrieved (reranked): ['Training Neural Networks - Gradient Descent and Backpropagation', 'Training Neural Networks - Gradient Descent and Backpropagation', 'Training Neural Networks - Gradient Descent and Backpropagation']  (recall=1.00, precision=1.00)
- retrieved (no rerank): ['Training Neural Networks - Gradient Descent and Backpropagation', 'Course Introduction and What Machine Learning Is', 'Training Neural Networks - Gradient Descent and Backpropagation']  (recall=1.00, precision=0.67)
- RAG answer: Backpropagation works by efficiently calculating the gradients of the loss function with respect to each weight in a neural network, layer by layer, starting from the output and moving backward through the layers. It uses the chain rule from calculus to compute these gradients. This process is crucial because without backpropagation, training very deep networks would be computationally impractical due to the sheer amount of calculations involved. Backpropagation allows for efficient computation of gradients, enabling the application of optimization algorithms like gradient descent to minimize the loss function and thus improve the model's performance.
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.00, contradiction_rate=0.00
  - **[FLAGGED LOW CONFIDENCE]**
- Baseline answer: Backpropagation is an algorithm used in training artificial neural networks to adjust weights by minimizing error between predicted outputs and actual targets. It works by calculating gradients of the loss function with respect to each weight, then updating these weights using those gradients. This process repeats iteratively until the network learns patterns from input data. Backpropagation is crucial because it enables efficient learning through gradient descent methods, allowing neural networks to improve their performance over time without manual tuning of parameters.
- Expected answer: Backpropagation computes the gradient of the loss with respect to every weight by working backwards from the output to the input using the chain rule, allowing gradient descent to update weights efficiently. Without it, training deep networks with millions of parameters would not be feasible.

### q4: What is dropout and how does it help prevent overfitting?
- relevant topic(s): ['Overfitting, Underfitting, and Regularization']
- retrieved (reranked): ['Overfitting, Underfitting, and Regularization', 'Overfitting, Underfitting, and Regularization', 'Overfitting, Underfitting, and Regularization']  (recall=1.00, precision=1.00)
- retrieved (no rerank): ['Overfitting, Underfitting, and Regularization', 'Overfitting, Underfitting, and Regularization', 'Overfitting, Underfitting, and Regularization']  (recall=1.00, precision=1.00)
- RAG answer: Dropout is a technique used to combat overfitting by randomly "switching off" a fraction of neurons during training. It helps prevent the network from relying too heavily on any single neuron, encouraging more robust, distributed representations. By doing so, dropout ensures that the model learns to generalize better to unseen data.
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.67, contradiction_rate=0.33
- Baseline answer: Dropout is a regularization technique used in neural networks to reduce overfitting by randomly setting a fraction of input units to zero during training, which helps improve generalization to new data. It works by forcing neurons to learn more robust features that are less likely to be affected by noise or irrelevant information.
- Expected answer: Dropout randomly switches off a fraction of neurons during each training forward pass, preventing the network from relying too heavily on any single neuron and encouraging more robust, distributed representations, which reduces overfitting.

### q5: What is the difference between a validation set and a test set?
- relevant topic(s): ['Overfitting, Underfitting, and Regularization']
- retrieved (reranked): ['Overfitting, Underfitting, and Regularization', 'Evaluating Machine Learning Models', 'Course Introduction and What Machine Learning Is']  (recall=1.00, precision=0.33)
- retrieved (no rerank): ['Evaluating Machine Learning Models', 'Overfitting, Underfitting, and Regularization', 'Evaluating Machine Learning Models']  (recall=1.00, precision=0.33)
- RAG answer: The validation set is used to tune hyperparameters such as the learning rate and to monitor for overfitting during training. The test set, on the other hand, is used exclusively at the end to get an honest estimate of how the model will perform in the real world.
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.00, contradiction_rate=0.50
  - **[FLAGGED LOW CONFIDENCE]**
- Baseline answer: A validation set is used to tune hyperparameters of an algorithm, while a test set evaluates its performance without overfitting to the training data.
- Expected answer: The validation set is used during training to tune hyperparameters and monitor for overfitting, while the test set is used only once at the end to get an honest estimate of real-world performance.

### q6: How do convolutional neural networks process images differently from a fully connected network?
- relevant topic(s): ['Convolutional Neural Networks for Image Processing']
- retrieved (reranked): ['Convolutional Neural Networks for Image Processing', 'Convolutional Neural Networks for Image Processing', 'Recurrent Neural Networks and Sequence Modelling']  (recall=1.00, precision=0.67)
- retrieved (no rerank): ['Convolutional Neural Networks for Image Processing', 'Convolutional Neural Networks for Image Processing', 'Convolutional Neural Networks for Image Processing']  (recall=1.00, precision=1.00)
- RAG answer: Convolutional Neural Networks (CNNs) process images differently from fully connected networks by focusing on local features within patches of the image rather than connecting all pixels directly to all neurons in subsequent layers. In CNNs, small "filters" or "kernels" move over the image, detecting simple features like edges and corners, which are then combined to form more complex structures. This approach allows CNNs to efficiently capture spatial hierarchies in images without needing to connect every pixel to every neuron, making them highly effective for image recognition tasks.
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.67, contradiction_rate=0.00
- Baseline answer: Convolutional Neural Networks (CNNs) process images by applying filters to small regions of an image, which helps in capturing spatial hierarchies and features like edges and textures. This is different from fully connected networks, which treat each pixel independently without considering its position relative to other pixels or higher-level structures within the image. CNNs use pooling layers to reduce dimensionality while preserving important features, whereas fully connected networks require all neurons to be interconnected with every input feature.
- Expected answer: CNNs use small filters/kernels that slide across the image to compute local features instead of connecting every pixel to every neuron, with early layers learning edges/corners and deeper layers combining these into complex shapes and objects; pooling layers reduce spatial size and add robustness to small translations.

### q7: What problem do LSTMs and GRUs solve compared to basic RNNs?
- relevant topic(s): ['Recurrent Neural Networks and Sequence Modelling']
- retrieved (reranked): ['Recurrent Neural Networks and Sequence Modelling', 'Recurrent Neural Networks and Sequence Modelling', 'Recurrent Neural Networks and Sequence Modelling']  (recall=1.00, precision=1.00)
- retrieved (no rerank): ['Recurrent Neural Networks and Sequence Modelling', 'Recurrent Neural Networks and Sequence Modelling', 'Recurrent Neural Networks and Sequence Modelling']  (recall=1.00, precision=1.00)
- RAG answer: LSTMs and GRUs solve the problem of vanishing gradients by introducing gating mechanisms within their architecture. These mechanisms allow the network to selectively keep or discard information from past time steps during training, thus preventing the gradients from becoming too small or exploding, which was a major issue with basic RNNs when dealing with long sequences.
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   1.00, contradiction_rate=0.00
- Baseline answer: LSTM (Long Short-Term Memory) networks and GRU (Gated Recurrent Unit) units address the vanishing gradient problem that affects standard RNNs, allowing them to maintain information over longer sequences without losing it as easily. This makes LSTM and GRU models more effective in tasks requiring long-term dependencies or sequence prediction.
- Expected answer: Basic RNNs suffer from the vanishing/exploding gradient problem over long sequences. LSTMs and GRUs use gating mechanisms to control what information is kept or discarded over time, helping the network retain useful information over longer sequences.

### q8: Why is accuracy alone not a good metric for imbalanced classification datasets, and what should be used instead?
- relevant topic(s): ['Evaluating Machine Learning Models']
- retrieved (reranked): ['Evaluating Machine Learning Models', 'Evaluating Machine Learning Models', 'Convolutional Neural Networks for Image Processing']  (recall=1.00, precision=0.67)
- retrieved (no rerank): ['Evaluating Machine Learning Models', 'Evaluating Machine Learning Models', 'Supervised, Unsupervised, and Reinforcement Learning']  (recall=1.00, precision=0.67)
- RAG answer: Accuracy alone is not a good metric for imbalanced classification datasets because it can be misleading when there are vastly different proportions of classes. In such cases, a high accuracy might indicate that the model is performing well overall but could be entirely due to its ability to predict the majority class. Instead, precision and recall are better suited to evaluate performance on imbalanced datasets by focusing on the quality of positive predictions rather than just their quantity.
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.00, contradiction_rate=0.00
  - **[FLAGGED LOW CONFIDENCE]**
- Baseline answer: Accuracy can be misleading in imbalanced datasets because it does not account for the imbalance of classes. A model might achieve high accuracy by simply predicting the majority class without considering minority instances. Instead, metrics like precision, recall, F1 score, or area under the ROC curve (AUC-ROC) are more appropriate as they provide a better understanding of how well the model performs across different types of errors.
- Expected answer: On imbalanced datasets a model can get high accuracy by always predicting the majority class while being useless. Precision, recall, the F1 score, and a confusion matrix give a more complete picture of performance.

### q9: What examples of bias in machine learning systems were discussed, and what other ethical concerns were raised?
- relevant topic(s): ['Ethics and Bias in Machine Learning']
- retrieved (reranked): ['Ethics and Bias in Machine Learning', 'Ethics and Bias in Machine Learning', 'Supervised, Unsupervised, and Reinforcement Learning']  (recall=1.00, precision=0.67)
- retrieved (no rerank): ['Ethics and Bias in Machine Learning', 'Ethics and Bias in Machine Learning', 'Supervised, Unsupervised, and Reinforcement Learning']  (recall=1.00, precision=0.67)
- RAG answer: The lecture discussed two main types of bias in machine learning systems:

1. **Societal Bias**: When machine learning models are trained on biased datasets, they can perpetuate or even exacerbate existing social inequalities. For instance, hiring algorithms that discriminate against women due to their training on historical data skewed towards men, or facial recognition systems that perform poorly on darker skin tones due to unrepresentative training data.

2. **Privacy Concerns**: Training machine learning models on personal data raises significant privacy issues. This includes potential misuse of sensitive information and the lack of control users have over how their data is used by AI systems.

Additionally, the lecture highlighted several broader ethical concerns beyond bias and privacy:

- **Transparency**: Modern deep learning models are often "black boxes" that make decisions difficult to understand, which is crucial in high-stakes applications such as healthcare or criminal justice.
  
- **Academic Integrity**: The risk of generating plausible-sounding but incorrect information, commonly referred to as "hall
  - faithfulness (concatenated)=0.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.38, contradiction_rate=0.00
- Baseline answer: Examples of bias in machine learning systems include gender biases in facial recognition software, racial biases in criminal justice algorithms, and age-related biases in hiring recommendations. Ethical concerns raised by these issues include fairness, accountability, transparency, and privacy.
- Expected answer: Examples include hiring algorithms that discriminated against women due to historical hiring data, and facial recognition systems that performed worse on darker-skinned faces due to unrepresentative training data. Other concerns raised include privacy, lack of transparency (black-box models), and the risk of AI systems generating plausible but incorrect information (hallucination).

### q10: Compare CNNs and RNNs in terms of the kind of data they are designed to process.
- relevant topic(s): ['Convolutional Neural Networks for Image Processing', 'Recurrent Neural Networks and Sequence Modelling']
- retrieved (reranked): ['Convolutional Neural Networks for Image Processing', 'Recurrent Neural Networks and Sequence Modelling', 'Convolutional Neural Networks for Image Processing']  (recall=1.00, precision=1.00)
- retrieved (no rerank): ['Recurrent Neural Networks and Sequence Modelling', 'Convolutional Neural Networks for Image Processing', 'Convolutional Neural Networks for Image Processing']  (recall=1.00, precision=1.00)
- RAG answer: CNNs are specifically designed for processing grid-like data such as images, whereas RNNs are designed to handle sequential data like text, audio, or time series.
  - faithfulness (concatenated)=1.00, contradiction_rate=0.00
  - faithfulness (per-chunk)=   0.00, contradiction_rate=0.00
  - **[FLAGGED LOW CONFIDENCE]**
- Baseline answer: CNNs are designed to process grid-like data such as images, where features tend to be spatially localized. They excel at identifying patterns and objects within these grids due to their convolutional layers that detect edges and textures. On the other hand, RNNs are better suited for sequential data like text or time series, where dependencies between elements need to be captured over time. They use recurrent connections to maintain state across different time steps, making them ideal for tasks involving sequence prediction or analysis.
- Expected answer: CNNs are designed for grid-like data such as images, using filters to capture spatial patterns. RNNs are designed for sequential data such as text, audio, or time series, maintaining a hidden state that carries information from earlier elements in the sequence.
