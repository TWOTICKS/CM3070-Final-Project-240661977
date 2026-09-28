# Cross-Domain Generalisation Evaluation

The same evaluation methodology as the main evaluation (Appendix A) run across three lecture domains with no shared vocabulary: the original evaluated ML lecture, a Roman History lecture, and a Nutrition Science lecture. This tests whether retrieval and faithfulness scoring generalise beyond the ML domain, or were implicitly tuned to its phrasing.

## Comparison across domains

| Domain | Recall@3 | Precision@3 | Faithfulness (per-chunk) | Contradiction (per-chunk) | Low-confidence flags | Out-of-scope abstained |
|---|---|---|---|---|---|---|
| ML Lecture (original, evaluated corpus) | 1.00 | 0.80 | 0.33 | 0.08 | 4/10 | False |
| Roman History | 1.00 | 0.70 | 0.54 | 0.05 | 1/10 | True |
| Nutrition Science | 1.00 | 0.67 | 0.49 | 0.06 | 2/10 | True |

## ML Lecture (original, evaluated corpus)

Questions: 11 (10 in-scope, 1 out-of-scope)

**Retrieval (mean recall@3 / precision@3, in-scope questions)**

- With re-ranking:    recall=1.00, precision=0.80
- Without re-ranking: recall=1.00, precision=0.80

**Faithfulness (mean over in-scope RAG answers)**

- Concatenated premise: faithfulness=0.10, contradiction=0.00
- Per-chunk premise:    faithfulness=0.33, contradiction=0.08

**Out-of-scope question(s)**

- Q: What did the lecturer say about quantum computing and its impact on machine learning?
  - top-3 rerank scores: [0.71, -7.34, -9.66]
  - abstained: False
  - RAG answer: The lecture excerpt does not mention anything about quantum computing and its impact on machine learning.
  - Baseline answer: The lecturer emphasized that quantum computing has the potential to revolutionize machine learning by potentially solving complex problems much faster than classical computers, especially in areas like optimization and cryptography. They also noted the ongoing research challenges and the need for interdisciplinary collaboration between computer scientists, physicists, and data scientists to fully realize this technology's benefits.

**Per-question detail (in-scope)**

- **q1**: What is the difference between supervised, unsupervised, and reinforcement learning?
  - retrieved: ['Supervised, Unsupervised, and Reinforcement Learning', 'Supervised, Unsupervised, and Reinforcement Learning', 'Supervised, Unsupervised, and Reinforcement Learning'] (recall=1.00, precision=1.00)
  - RAG answer: Supervised learning involves training models with labeled data, aiming to map inputs to correct outputs. Unsupervised learning finds patterns in unlabeled data, grouping similar items together. Reinforcement learning teaches agents to maximize rewards through interaction with environments, optimizing strategies over repeated trials.
  - faithfulness (concat)=0.00, (per-chunk)=0.33, contradiction (per-chunk)=0.00
- **q2**: What is the ReLU activation function and why is it popular in deep learning?
  - retrieved: ['Neural Network Fundamentals', 'Neural Network Fundamentals', 'Training Neural Networks - Gradient Descent and Backpropagation'] (recall=1.00, precision=0.67)
  - RAG answer: The ReLU (Rectified Linear Unit) activation function is used in neural networks to introduce non-linearity. It outputs zero for any negative input and passes the absolute value of the input through for positive inputs. This makes it computationally efficient and helps prevent vanishing gradient problems common in deep networks with sigmoid activations. Its popularity stems from its simplicity, computational efficiency, and effectiveness in avoiding issues like exploding gradients.
  - faithfulness (concat)=0.00, (per-chunk)=0.25, contradiction (per-chunk)=0.00
- **q3**: How does backpropagation work and why is it important for training neural networks?
  - retrieved: ['Training Neural Networks - Gradient Descent and Backpropagation', 'Training Neural Networks - Gradient Descent and Backpropagation', 'Training Neural Networks - Gradient Descent and Backpropagation'] (recall=1.00, precision=1.00)
  - RAG answer: Backpropagation works by efficiently calculating the gradients of the loss function with respect to each weight in a neural network, layer by layer, starting from the output and moving backward through the layers. It uses the chain rule from calculus to compute these gradients. This process is crucial because without backpropagation, training very deep networks would be computationally impractical due to the sheer amount of calculations involved. Backpropagation allows for efficient computation of gradients, enabling the application of optimization algorithms like gradient descent to minimize the loss function and thus improve the model's performance.
  - faithfulness (concat)=0.00, (per-chunk)=0.00, contradiction (per-chunk)=0.00
- **q4**: What is dropout and how does it help prevent overfitting?
  - retrieved: ['Overfitting, Underfitting, and Regularization', 'Overfitting, Underfitting, and Regularization', 'Overfitting, Underfitting, and Regularization'] (recall=1.00, precision=1.00)
  - RAG answer: Dropout is a technique used to combat overfitting by randomly "switching off" a fraction of neurons during training. It helps prevent the network from relying too heavily on any single neuron, encouraging more robust, distributed representations. By doing so, dropout ensures that the model learns to generalize better to unseen data.
  - faithfulness (concat)=0.00, (per-chunk)=0.67, contradiction (per-chunk)=0.33
- **q5**: What is the difference between a validation set and a test set?
  - retrieved: ['Overfitting, Underfitting, and Regularization', 'Evaluating Machine Learning Models', 'Course Introduction and What Machine Learning Is'] (recall=1.00, precision=0.33)
  - RAG answer: The validation set is used to tune hyperparameters such as the learning rate and to monitor for overfitting during training. The test set, on the other hand, is used exclusively at the end to get an honest estimate of how the model will perform in the real world.
  - faithfulness (concat)=0.00, (per-chunk)=0.00, contradiction (per-chunk)=0.50
- **q6**: How do convolutional neural networks process images differently from a fully connected network?
  - retrieved: ['Convolutional Neural Networks for Image Processing', 'Convolutional Neural Networks for Image Processing', 'Recurrent Neural Networks and Sequence Modelling'] (recall=1.00, precision=0.67)
  - RAG answer: Convolutional Neural Networks (CNNs) process images differently from fully connected networks by focusing on local features within patches of the image rather than connecting all pixels directly to all neurons in subsequent layers. In CNNs, small "filters" or "kernels" move over the image, detecting simple features like edges and corners, which are then combined to form more complex structures. This approach allows CNNs to efficiently capture spatial hierarchies in images without needing to connect every pixel to every neuron, making them highly effective for image recognition tasks.
  - faithfulness (concat)=0.00, (per-chunk)=0.67, contradiction (per-chunk)=0.00
- **q7**: What problem do LSTMs and GRUs solve compared to basic RNNs?
  - retrieved: ['Recurrent Neural Networks and Sequence Modelling', 'Recurrent Neural Networks and Sequence Modelling', 'Recurrent Neural Networks and Sequence Modelling'] (recall=1.00, precision=1.00)
  - RAG answer: LSTMs and GRUs solve the problem of vanishing gradients by introducing gating mechanisms within their architecture. These mechanisms allow the network to selectively keep or discard information from past time steps during training, thus preventing the gradients from becoming too small or exploding, which was a major issue with basic RNNs when dealing with long sequences.
  - faithfulness (concat)=0.00, (per-chunk)=1.00, contradiction (per-chunk)=0.00
- **q8**: Why is accuracy alone not a good metric for imbalanced classification datasets, and what should be used instead?
  - retrieved: ['Evaluating Machine Learning Models', 'Evaluating Machine Learning Models', 'Convolutional Neural Networks for Image Processing'] (recall=1.00, precision=0.67)
  - RAG answer: Accuracy alone is not a good metric for imbalanced classification datasets because it can be misleading when there are vastly different proportions of classes. In such cases, a high accuracy might indicate that the model is performing well overall but could be entirely due to its ability to predict the majority class. Instead, precision and recall are better suited to evaluate performance on imbalanced datasets by focusing on the quality of positive predictions rather than just their quantity.
  - faithfulness (concat)=0.00, (per-chunk)=0.00, contradiction (per-chunk)=0.00
- **q9**: What examples of bias in machine learning systems were discussed, and what other ethical concerns were raised?
  - retrieved: ['Ethics and Bias in Machine Learning', 'Ethics and Bias in Machine Learning', 'Supervised, Unsupervised, and Reinforcement Learning'] (recall=1.00, precision=0.67)
  - RAG answer: The lecture discussed two main types of bias in machine learning systems:

1. **Societal Bias**: When machine learning models are trained on biased datasets, they can perpetuate or even exacerbate existing social inequalities. For instance, hiring algorithms that discriminate against women due to their training on historical data skewed towards men, or facial recognition systems that perform poorly on darker skin tones due to unrepresentative training data.

2. **Privacy Concerns**: Training machine learning models on personal data raises significant privacy issues. This includes potential misuse of sensitive information and the lack of control users have over how their data is used by AI systems.

Additionally, the lecture highlighted several broader ethical concerns beyond bias and privacy:

- **Transparency**: Modern deep learning models are often "black boxes" that make decisions difficult to understand, which is crucial in high-stakes applications such as healthcare or criminal justice.
  
- **Academic Integrity**: The risk of generating plausible-sounding but incorrect information, commonly referred to as "hall
  - faithfulness (concat)=0.00, (per-chunk)=0.38, contradiction (per-chunk)=0.00
- **q10**: Compare CNNs and RNNs in terms of the kind of data they are designed to process.
  - retrieved: ['Convolutional Neural Networks for Image Processing', 'Recurrent Neural Networks and Sequence Modelling', 'Convolutional Neural Networks for Image Processing'] (recall=1.00, precision=1.00)
  - RAG answer: CNNs are specifically designed for processing grid-like data such as images, whereas RNNs are designed to handle sequential data like text, audio, or time series.
  - faithfulness (concat)=1.00, (per-chunk)=0.00, contradiction (per-chunk)=0.00

## Roman History

Questions: 11 (10 in-scope, 1 out-of-scope)

**Retrieval (mean recall@3 / precision@3, in-scope questions)**

- With re-ranking:    recall=1.00, precision=0.70
- Without re-ranking: recall=1.00, precision=0.67

**Faithfulness (mean over in-scope RAG answers)**

- Concatenated premise: faithfulness=0.21, contradiction=0.07
- Per-chunk premise:    faithfulness=0.54, contradiction=0.05

**Out-of-scope question(s)**

- Q: What did the lecturer say about dropout and neural network regularization?
  - top-3 rerank scores: [-11.37, -11.38, -11.38]
  - abstained: True
  - RAG answer: I couldn't find anything in this lecture covering that topic.
  - Baseline answer: The lecturer emphasized that dropout is an effective technique to prevent overfitting in neural networks by randomly dropping out units during training, which helps generalize better. Regularization techniques like L1 or L2 were also discussed as another method to control model complexity and improve generalization.

**Per-question detail (in-scope)**

- **r1**: What legendary story explains the founding of Rome, and how did the monarchy end?
  - retrieved: ['The Founding and Early Kingdom of Rome', 'The Founding and Early Kingdom of Rome', 'The Decline and Fall of the Western Empire'] (recall=1.00, precision=0.67)
  - RAG answer: The legend of Rome's founding comes from tradition, claiming that Romulus and Remus, twins abandoned as infants, were raised by a she-wolf and later founded the city. According to this tale, Romulus killed Remus in a dispute over the location of the new settlement, becoming its first king. However, archaeological evidence suggests that the area was inhabited long before any single founding date, indicating a gradual merging of settlements rather than a singular event. The traditional account of the monarchy's end is through the overthrow of King Tarquin the Proud in 509 BCE, marking the abolition of the monarchy.
  - faithfulness (concat)=0.00, (per-chunk)=0.75, contradiction (per-chunk)=0.00
- **r2**: How did the Roman Republic divide power between the Senate, consuls, and tribunes to prevent tyranny?
  - retrieved: ['The Roman Republic and Its Institutions', 'The Roman Republic and Its Institutions', 'Augustus and the Beginning of the Empire'] (recall=1.00, precision=0.67)
  - RAG answer: The Roman Republic divided power through several mechanisms to prevent tyranny:

1. **Senate**: Made up of wealthy landowners called patricians, advising on policy and controlling finances.
2. **Consuls**: Elected annually with veto powers over each other, providing a check against tyranny.
3. **Magistrates**: Various offices handling different aspects of governance, including the treasury, courts, and city administration.
4. **Plebeians' Assembly**: Gained the right to elect officials called tribunes, who could veto actions by the Senate or magistrates that harmed plebeian interests.

These structures collectively ensured checks and balances among different groups within the state, preventing any single group from gaining excessive control.
  - faithfulness (concat)=0.22, (per-chunk)=0.33, contradiction (per-chunk)=0.00
- **r3**: What happened during the Second Punic War, and how did Rome ultimately defeat Carthage?
  - retrieved: ['The Punic Wars and Roman Expansion', 'The Punic Wars and Roman Expansion', 'The Decline and Fall of the Western Empire'] (recall=1.00, precision=0.67)
  - RAG answer: During the Second Punic War, Hannibal led a massive Carthaginian army through the Alps into Italy, inflicting significant defeats on Roman forces. Despite these setbacks, Rome refused to give up and rebuilt its military strength. Eventually, General Scipio Africanus defeated Hannibal at the Battle of Zama in 202 BCE, decisively ending the conflict.
  - faithfulness (concat)=0.00, (per-chunk)=0.67, contradiction (per-chunk)=0.00
- **r4**: Why is Julius Caesar's crossing of the Rubicon significant, and how did he die?
  - retrieved: ['Julius Caesar and the Fall of the Republic', 'Julius Caesar and the Fall of the Republic', 'Augustus and the Beginning of the Empire'] (recall=1.00, precision=0.67)
  - RAG answer: Julius Caesar's crossing of the Rubicon River marked a critical moment where he declared war on the Roman Republic, effectively ending its existence. This act symbolized his ambition to become dictatorial powers over Rome, leading to his eventual assassination by a group of senators on March 15, 44 BCE.
  - faithfulness (concat)=0.00, (per-chunk)=0.00, contradiction (per-chunk)=0.00
- **r5**: How did Augustus establish the Principate while appearing to preserve republican government?
  - retrieved: ['Augustus and the Beginning of the Empire', 'Augustus and the Beginning of the Empire', 'The Roman Republic and Its Institutions'] (recall=1.00, precision=0.67)
  - RAG answer: Augustus established the Principate by presenting himself as "princeps" or first citizen, effectively becoming the sole ruler of the Roman Empire through a combination of constitutional reforms and propaganda. He received the title "Augustus" and gradually gained control over the state apparatus, including the Senate and the army. By doing so, he maintained appearances of republican governance while consolidating his power.
  - faithfulness (concat)=0.00, (per-chunk)=0.33, contradiction (per-chunk)=0.33
- **r6**: What Roman engineering achievements allowed the empire to function as one connected system?
  - retrieved: ['Roman Engineering and Infrastructure', 'Roman Engineering and Infrastructure', 'Augustus and the Beginning of the Empire'] (recall=1.00, precision=0.67)
  - RAG answer: Roman engineering, including their extensive road networks, aqueducts, and concrete structures, allowed the empire to function as one connected system by facilitating rapid movement of people, goods, and military forces across vast distances. These infrastructural advancements helped maintain political control and cultural unity within the sprawling Roman Empire.
  - faithfulness (concat)=0.50, (per-chunk)=1.00, contradiction (per-chunk)=0.00
- **r7**: What was daily life like for different social classes in ancient Rome?
  - retrieved: ['Daily Life in Ancient Rome', 'Daily Life in Ancient Rome', 'The Founding and Early Kingdom of Rome'] (recall=1.00, precision=0.67)
  - RAG answer: In ancient Rome, daily life varied significantly across different social classes:

1. **Senators and Equestrians**: These elite groups enjoyed high status and wealth. They lived in luxurious homes and participated in public affairs. Their lives were marked by grand ceremonies and political activities.

2. **Ordinary Free Citizens**: This class comprised most of the population but faced significant limitations. They lived in crowded, sometimes dangerous, multi-story buildings known as insulae. Daily life centered around basic sustenance such as bread, olive oil, wine, and vegetables, with meat being a rare luxury item.

3. **Freedmen**: Formerly enslaved individuals gained freedom through legal means and could rise socially. Their lives were more stable than those of slaves but still constrained by their former servitude.

4. **Enslaved Population**: The vast majority of the population was enslaved, providing labor essential to the economy. Their lives were harsh and exploitative, serving as a crucial part of the economic system.

The
  - faithfulness (concat)=0.00, (per-chunk)=0.27, contradiction (per-chunk)=0.07
- **r8**: What economic, political, and military factors led to the fall of the Western Roman Empire?
  - retrieved: ['The Decline and Fall of the Western Empire', 'The Decline and Fall of the Western Empire', 'Augustus and the Beginning of the Empire'] (recall=1.00, precision=0.67)
  - RAG answer: The Western Roman Empire fell due to multiple interconnected factors including chronic inflation, heavy taxation, and a shrinking tax base; frequent civil wars and a rapid succession of emperors leading to weak central authority; and sustained pressure from external threats such as invasions from northern and eastern fronts, coupled with reliance on foreign mercenaries who lacked loyalty to Rome.
  - faithfulness (concat)=1.00, (per-chunk)=1.00, contradiction (per-chunk)=0.00
- **r9**: How does Roman law and Republican institutional design influence modern government?
  - retrieved: ['The Legacy of Rome', 'The Legacy of Rome', 'The Roman Republic and Its Institutions'] (recall=1.00, precision=0.67)
  - RAG answer: Roman law and Republican institutional designs have significantly influenced modern governance through several key aspects:

1. Legal structures: Roman law forms the basis for legal systems in much of Europe and has deeply impacted legal procedures and contract laws.
2. Language evolution: Latin gave rise to Romance languages such as French, Spanish, Italian, Portuguese, and Romanian, influencing English vocabulary, particularly technical and formal terms.
3. Architectural elements: Roman architectural styles, including the arch, dome, and use of concrete, continue to be used in modern civic architecture.
4. Institutional models: The concept of checks and balances, elected representatives, and term limits found in Roman republican institutions have been adopted by modern constitutions, including those of the United States.
  - faithfulness (concat)=0.00, (per-chunk)=0.62, contradiction (per-chunk)=0.12
- **r10**: Compare the checks on power in the Roman Republic with the authority Augustus actually held under the Principate.
  - retrieved: ['Augustus and the Beginning of the Empire', 'The Roman Republic and Its Institutions', 'Augustus and the Beginning of the Empire'] (recall=1.00, precision=1.00)
  - RAG answer: In the Roman Republic, power was divided among different institutions to prevent any single individual from gaining excessive control. The Senate advised on policies but lacked formal legislative powers; executive authority was shared between two consuls, who could veto each other. Various magistrates managed specific areas such as finance, justice, and local governance. Ordinary citizens had limited rights compared to the elite class, and they only gained the right to elect tribunes who could veto Senate decisions affecting their interests. This system aimed to balance power through checks and balances.

Under the Principate established by Augustus, power was concentrated more heavily in the hands of a single ruler. He was referred to as "Augustus" and presented himself as "princeps," or first citizen, rather than king. His position was formally recognized when the Senate bestowed upon him the title "Augustus." While he retained significant influence over military and administrative matters, he also took on roles traditionally associated with kingship, such as being considered divine. This shift represented
  - faithfulness (concat)=0.33, (per-chunk)=0.44, contradiction (per-chunk)=0.00

## Nutrition Science

Questions: 11 (10 in-scope, 1 out-of-scope)

**Retrieval (mean recall@3 / precision@3, in-scope questions)**

- With re-ranking:    recall=1.00, precision=0.67
- Without re-ranking: recall=1.00, precision=0.60

**Faithfulness (mean over in-scope RAG answers)**

- Concatenated premise: faithfulness=0.07, contradiction=0.00
- Per-chunk premise:    faithfulness=0.49, contradiction=0.06

**Out-of-scope question(s)**

- Q: What did the lecturer say about the Roman Senate and the outcome of the Punic Wars?
  - top-3 rerank scores: [-10.98, -11.1, -11.12]
  - abstained: True
  - RAG answer: I couldn't find anything in this lecture covering that topic.
  - Baseline answer: The lecturer mentioned that the Roman Senate played a crucial role in managing foreign policy, including negotiations with Carthage during the Punic Wars. The Senate was instrumental in deciding to declare war against Carthage after its defeat at the Battle of Cannae, which led to Rome gaining control over Sicily and southern Italy. The Senate also had significant influence in negotiating peace terms with Carthage following the wars.

**Per-question detail (in-scope)**

- **n1**: What is nutrition science and how did it originally develop as a field?
  - retrieved: ['What Is Nutrition Science and Why It Matters', 'What Is Nutrition Science and Why It Matters', 'Sustainable Food Systems'] (recall=1.00, precision=0.67)
  - RAG answer: Nutrition science is the study of how the food we eat affects our bodies, including how it breaks down, absorbs, and uses nutrients to provide energy and building blocks for growth and maintenance. It originated historically through efforts to understand deficiency diseases caused by a lack of essential nutrients in the diet, such as scurvy due to vitamin C deficiency.
  - faithfulness (concat)=0.00, (per-chunk)=0.50, contradiction (per-chunk)=0.00
- **n2**: What are the three macronutrients and what does each provide the body?
  - retrieved: ['Macronutrients - Carbohydrates, Protein, and Fat', 'Micronutrients - Vitamins and Minerals', 'Macronutrients - Carbohydrates, Protein, and Fat'] (recall=1.00, precision=0.67)
  - RAG answer: The three macronutrients are carbohydrates, protein, and fat. Carbohydrates serve as the primary quick energy source, breaking down into glucose which cells use for immediate energy. They include both simple sugars absorbed quickly and complex starches and fiber that digest more slowly, affecting blood sugar levels gently. Protein builds tissues, produces enzymes and hormones, and supports the immune system, although the body stores less protein compared to carbohydrates or fat. Fat provides twice the calories per gram of carbohydrates and protein, making it essential for vitamin absorption, cell membrane formation, hormone production, and energy storage.
  - faithfulness (concat)=0.00, (per-chunk)=1.00, contradiction (per-chunk)=0.00
- **n3**: What is the difference between fat-soluble and water-soluble vitamins?
  - retrieved: ['Micronutrients - Vitamins and Minerals', 'Macronutrients - Carbohydrates, Protein, and Fat', 'Macronutrients - Carbohydrates, Protein, and Fat'] (recall=1.00, precision=0.33)
  - RAG answer: Fat-soluble vitamins, such as A, D, E, and K, can be stored in the body's fatty tissues. Water-soluble vitamins, including vitamin C and the B vitamins, cannot be stored and must be regularly replenished through diet.
  - faithfulness (concat)=0.00, (per-chunk)=0.00, contradiction (per-chunk)=0.50
- **n4**: What food preservation techniques were discussed and how do they prevent spoilage?
  - retrieved: ['Food Preservation Techniques', 'Food Preservation Techniques', 'Sustainable Food Systems'] (recall=1.00, precision=0.67)
  - RAG answer: The lecture covered several food preservation techniques that prevent spoilage:

1. **Drying**: Removes moisture needed for bacterial growth, preserving foods like dried fruits, jerky, and grains.
2. **Salting and Sugaring**: Draw water out of food through osmosis, extending shelf life of preserved items such as cured meats and jams.
3. **Refrigeration and Freezing**: Slow down chemical and microbial processes causing spoilage but do not completely halt them, leading to eventual degradation of frozen food.
4. **Canning**: Heats food to kill microorganisms and seals it in containers to prevent recontamination, developed for feeding Napoleon’s army.
5. **Pasteurization**: Briefly heats food to kill pathogens without fully cooking it.
6. **Vacuum Sealing**: Removes oxygen necessary for some spoilage organisms to survive.

These methods collectively help extend the storage time of food while maintaining its nutritional value and safety.
  - faithfulness (concat)=0.00, (per-chunk)=0.15, contradiction (per-chunk)=0.08
- **n5**: What is the Maillard reaction and why does it matter for cooking?
  - retrieved: ['The Maillard Reaction and Cooking Chemistry', 'The Maillard Reaction and Cooking Chemistry', 'Fermentation and the Role of Microorganisms'] (recall=1.00, precision=0.67)
  - RAG answer: The Maillard reaction is a chemical reaction between amino acids and reducing sugars that occurs during heating, particularly above 140°C. It results in the production of numerous flavor and aroma compounds, making browned food taste significantly more complex compared to uncooked food. This reaction is crucial for processes such as searing meats, toasting bread, and roasting coffee beans, contributing to the rich flavors associated with these culinary methods.
  - faithfulness (concat)=0.00, (per-chunk)=0.67, contradiction (per-chunk)=0.00
- **n6**: What is fermentation and what foods does it produce?
  - retrieved: ['Fermentation and the Role of Microorganisms', 'Fermentation and the Role of Microorganisms', 'Sustainable Food Systems'] (recall=1.00, precision=0.67)
  - RAG answer: Fermentation is a metabolic process involving microorganisms like bacteria, yeast, or mold breaking down sugars and other compounds in food. It results in the production of acids, gases, or alcohol as byproducts. Common examples include bread rising due to yeast fermentation and the creation of alcoholic beverages such as beer and wine. Additionally, lactic acid fermentation is used in making yogurt, sauerkraut, kimchi, and traditional pickled vegetables.
  - faithfulness (concat)=0.25, (per-chunk)=0.75, contradiction (per-chunk)=0.00
- **n7**: What are the four core principles of food safety?
  - retrieved: ['Food Safety and Foodborne Illness', 'Food Safety and Foodborne Illness', 'What Is Nutrition Science and Why It Matters'] (recall=1.00, precision=0.67)
  - RAG answer: The four core principles of food safety summarized in the lectures are:

1. **Clean**: Wash hands and surfaces to prevent contamination.
2. **Separate**: Avoid cross-contamination by using different cutting boards for raw foods and ready-to-eat items.
3. **Cook**: Heat food to a safe internal temperature to kill relevant pathogens.
4. **Chill**: Refrigerate perishable food promptly to slow bacterial growth.
  - faithfulness (concat)=0.00, (per-chunk)=0.00, contradiction (per-chunk)=0.00
- **n8**: How do nutritional needs change across different life stages?
  - retrieved: ['Nutrition Across the Life Course', 'Nutrition Across the Life Course', 'What Is Nutrition Science and Why It Matters'] (recall=1.00, precision=0.67)
  - RAG answer: Nutritional needs change significantly across different life stages. In infancy, breast milk or formula provides essential nutrition until solids can be introduced gradually. During childhood and adolescence, there is increased demand for proteins, calcium, and iron due to rapid growth. As adulthood approaches, nutritional needs focus on maintaining current health while also preventing diseases. During pregnancy, additional nutrients such as folate, iron, and calories are required to support fetal development. In later life, although calorie needs may decrease slightly, the need for certain nutrients like protein, calcium, and vitamin D remains constant or increases, posing challenges for older adults who must balance these needs with reduced caloric intake.
  - faithfulness (concat)=0.00, (per-chunk)=0.67, contradiction (per-chunk)=0.00
- **n9**: Why is food production considered an environmental issue?
  - retrieved: ['Sustainable Food Systems', 'Sustainable Food Systems', 'Fermentation and the Role of Microorganisms'] (recall=1.00, precision=0.67)
  - RAG answer: Food production is considered an environmental issue due to its significant impact on greenhouse gas emissions, land use changes, and freshwater consumption. Producing a kilogram of beef typically requires much more land and water compared to plant-based protein sources, leading to higher emissions per unit of food produced. Additionally, food waste represents wasted resources throughout the entire production chain, further exacerbating environmental concerns.
  - faithfulness (concat)=0.00, (per-chunk)=0.67, contradiction (per-chunk)=0.00
- **n10**: How does a deficiency in a specific micronutrient connect to the historical origins of nutrition science?
  - retrieved: ['What Is Nutrition Science and Why It Matters', 'Micronutrients - Vitamins and Minerals', 'What Is Nutrition Science and Why It Matters'] (recall=1.00, precision=1.00)
  - RAG answer: A deficiency in a specific micronutrient connects to the historical origins of nutrition science through the understanding of deficiency diseases such as scurvy and rickets, which were historically caused by a lack of certain nutrients in the diet. These early studies laid the foundation for recognizing the importance of vitamins and minerals in maintaining human health.
  - faithfulness (concat)=0.50, (per-chunk)=0.50, contradiction (per-chunk)=0.00
