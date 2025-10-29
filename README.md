 Abstract
 
 Accurate power estimation is essential for optimizing digital circuit design in modern VLSI systems. This study employs machine learning techniques, specifically ridge regression and linear regression models, to predict power consumption based on circuit characteristics. The dataset includes various digital circuits with key parameters such as the number of cells, I/O ports, nets, area, and multiple power metrics, including leakage, internal, net, dynamic switching, and total power. Feature analysis is conducted to identify the most significant contributors to power consumption. The proposed ridge regression-based approach outperforms linear regression by effectively handling multicollinearity and providing more stable predictions. Compared to traditional estimation methods, ridge regression demonstrates high prediction accuracy and computational efficiency, making it a suitable choice for realtime power analysis. The results indicate that machine learning, particularly regularized regression techniques, can significantly enhance power-aware design automation, enabling faster and more efficient optimization of digital circuits. The analysis of power prediction accuracy highlights Ridge Regression, Linear Regression, and CatBoost as the top-performing models, achieving exceptional accuracies of 99.998864%, 99.998706%, and 99.993789%, respectively. These models demonstrate outstanding precision in estimating total power, with Ridge Regression showing the lowest absolute error of 1.437 nW, followed closely by Linear Regression and CatBoost. Such high levels of accuracy confirm their reliability and suitability for power estimation tasks, making them ideal choices for applications demanding precise and dependable predictions.

 INTRODUCTION

  The increasing complexity of modern VLSI circuits has made power consumption a critical design constraint.High power dissipation results in high thermal effects, low device reliability, and high operating cost. Thus, efficient estimation methodologies of power are required in order to enhance the efficiency of the circuit with reduced energy consumption. Mathematical formulation-based and simulation methodologybased power analysis techniques are computationally expensive, although they are accurate, and hence are unfeasible in large circuits. These constraints impose the necessity of methodologies that estimate the power with high accuracy but with low computational overhead.
  
  Machine learning has emerged as a promising technique for power estimation by leveraging data-driven models to learn complex relationships between circuit parameters and power consumption. Among these, the number of cells and circuit area are two critical factors that significantly impact power dissipation. Understanding their influence on power consumption can help optimize circuit design, leading to improved efficiency and reduced energy usage.
 
 The cell count in a digital circuit is instrumental in determining the overall power consumption of the circuit. A cell in a circuit is basically a logic building block, and the more the cell count, the more the power consumption during operation. The main causes of this added power consumption are leakage power consumption, internal dissipation of power, and dynamic power consumption due to switching activity. More cell counts translate into more transistor density, which translates into more leakage current that contributes to overall consumption even when the circuit is in an inactive mode. Additionally, internal dissipation of power caused by shortcircuit current as well as capacitive charging within logic cells rises with the increase in cell counts. A significant contributor to the consumption of power is also the circuit area. Power dissipation in circuits is physically affected in many ways, most predominantly through signal transition delay and interconnect power. Larger circuit areas bring with them longer interconnects, with the effect being more parasitic capacitances and more net consumption of power. The incremental boost in terms of power is because of more resistance and capacitance in longer wires, causing more dissipation of energy in signal transitions. Furthermore, as the circuit area is bigger, signal integrity is more difficult to maintain, often mandating more buffering, which is another incremental contributor to consumption of power. Reducing the circuit area through efficient design methodology is thus beneficial in reducing dissipation of power, and is a significant parameter in terms of poweraware optimization.

 METHODOLOGY

 Power estimation in digital circuits needs an efficient and precise methodology because of the growing VLSI design complexities. The current work focuses on examining some machine learning models in order to estimate the power consumption from the circuit parameters. The framework is proposed as indicated in Fig 1.Ultimately, after comparing various models, the final methodology chosen was the application of Ridge Regression due to its better predictive capability. The methodology includes the dataset preprocessing, selection of the model, hyperparameter tuning, and the evaluation such that the framework is optimised and trustworthy in terms of estimating the power. 

 ![block diagram](https://github.com/user-attachments/assets/b09bce5b-55b1-49b5-9f6b-eba76ecd6ab6)

 Fig 1. Block Diagram of Proposed Model

 A. Dataset Description and Preprocessing

  The dataset consists of approximately 300 unique digital circuit designs, encompassing a diverse set of architectures including arithmetic blocks, memory circuits, and control units implemented. Each data record contains parameters such as number of cells, number of I/O ports, nets, circuit area, leakage power, internal power, net power, and the target values: total and dynamic power. To ensure consistency and improve model performance, data preprocessing was performed, including missing value handling, normalization, and feature selection. Since raw data can have different scales, normalization techniques such as min-max scaling were applied to standardize feature values. Additionally, outlier detection was conducted to remove anomalies that could impact model training.

  The dataset was randomly split as 80% for training, 20% for testing. This diversity ensures the generalizability of the proposed models to a wide range of digital designs.


IMPLEMENTATION

 A. Machine Learning Models Evaluation

 The flow chart of implementaion is given in Fig.2 and several machine learning models were explored for power prediction, each with its advantages and limitations. The models evaluated include:

 • Linear Regression: Used as a baseline model to establish a simple relationship between circuit parameters and power consumption. However, it failed to capture nonlinear dependencies, leading to suboptimal accuracy.

 • Ridge regression: linear model that works well with complex and correlated datasets because it uses L2 regularization to reduce overfitting, manage multicollinearity, and enhance generalization by reducing large coefficients.

 • Decision Trees: Provided better interpretability and handled feature interactions well. However, the model exhibited overfitting on training data, reducing its generalizability.

 • Random Forest: Improved upon decision trees by averaging multiple trees to reduce overfitting. Although it provided better accuracy than standalone decision trees, its performance was inconsistent across different circuit types.

 • Gradient Boosting Machines (GBM): Implemented to enhance predictive performance by iteratively reducing errors. GBM demonstrated strong results but required extensive hyperparameter tuning to prevent overfitting.

 • Lasso Regression: Used for feature selection by shrinking less important coefficients to zero, improving model interpretability and reducing overfitting. It provided stable predictions but at the cost of slightly higher bias. 

 • LightGBM: Designed for efficiency and speed, utilizing histogram-based learning to handle large datasets. It achieved fast training times but required careful tuning to avoid overfitting.

 • XGBoost: Optimized for performance with parallel processing and regularization techniques. It delivered high accuracy but demanded significant computational resources and hyperparameter tuning.

 • Support Vector Regression (SVR): Evaluated for its ability to model nonlinear relationships using kernel functions. SVR showed competitive performance but had high computational complexity, making it less efficient for large datasets.

 • Multi-Layer Perceptron (MLP): A deep learning approach capable of capturing complex relationships between circuit parameters and power consumption. The model was trained with multiple hidden layers and optimized activation functions, resulting in high prediction accuracy.

 <img width="423" height="596" alt="image" src="https://github.com/user-attachments/assets/432625e0-6e9f-4aa2-af09-e29a2bda4946" />

 Fig. 2. Flow Chart of Implementation

 B. Model Selection and Parameter Optimization

 To determine the most suitable model, performance metrics such as mean absolute error (MAE), mean squared error (MSE), and R squared (R2) were used for evaluation. Among all models, Ridge Regression exhibited the best predictive accuracy, effectively capturing nonlinear dependencies in power consumption data.

 <img width="509" height="330" alt="image" src="https://github.com/user-attachments/assets/65abb176-f910-4f65-91ad-c067305d7bb6" />

 The Table I compares the actual and predicted dynamic power values. The actual dynamic power represents the real measured power consumption, while the predicted dynamic power is the estimated value provided by each model. Most models predict values close to the actual dynamic power, but some, such as the MLP Regressor and LightGBM, show significant deviations. This indicates that these models may have overfitting issues or struggle with generalization. However, models like linear regression, ridge regression, and XGBoost provide predictions very close to the actual values, suggesting better accuracy and reliability.

 Each model’s hyperparameters were optimized using grid and random search strategies. For instance, MLP used two hidden layers with 64 and 32 neurons respectively, ReLU activation, and Adam optimizer with a learning rate of 0.01. Ridge Regression’s alpha was set to 0.05 based on validation performance. This information ensures the study reproducible. Overfitting, a common challenge in machine learning especially with complex models like neural networks and ensemble methods, was addressed through multiple strategies in this study. Ridge Regression inherently uses L2 regularization to control coefficient magnitude and manage multicollinearity. Additionally, k-fold cross-validation was employed to ensure robust generalization across different data splits. Hyperparameter optimization via grid and random search fine-tuned parameters such as regularization strength, network architecture, dropout rates, and tree depths to balance model complexity and fit. Feature selection and normalization reduced noise and irrelevant variability, while dropout layers in neural networks enhanced robustness by randomly deactivating neurons during training. These combined techniques resulted in stable and accurate power predictions, with Ridge Regression notably minimizing overfitting and delivering consistent performance across diverse circuit datasets.

  C. Model Evaluation and Validation

  The dataset is giving good results for models of Ridge regression and linear regression and it is avoiding overfitting of data which usually happens in MLP for limited dataset and the actual and predicted power plot is given in Fig 3 for some set of data in the dataset. The sophisticated ensemble models like XGBoost and CatBoost tend to underperform because they are designed to capture intricate, non-linear feature interactions and are more prone to overfitting when faced with moderatesized, low-noise datasets dominated by linear patterns. Despite extensive hyperparameter tuning, both XGBoost and CatBoost demonstrated a tendency to slightly overfit the data, resulting in higher mean absolute error (MAE) and mean squared error (MSE) than expected for this regression task. Conversely, Ridge Regression is specifically tailored for scenarios with correlated features and linear dependencies. Its L2 regularization actually suppresses multicollinearity because large coefficient penalties yield stable and highly accurate forecasts. As the structure and characteristics of the dataset well align with the Ridge Regression benefits, this model always generates higher generalizability and predictive ability, obviously as being the most appropriate option in terms of efficienct power estimation during digital circuit design.

  Strong and consistent power estimates were provided through the findings, which demonstrated stable performance with multiple subsets and verified the model’s resilience against multicollinearity especially through Ridge regression.

  <img width="470" height="315" alt="image" src="https://github.com/user-attachments/assets/68fa67d2-f534-4fcd-9b58-848747b61bfd" />

  Fig. 3. Prediction vs Actual plot

  RESULTS

  The performance of various machine learning models for power prediction in VLSI circuits was evaluated using a dataset containing power consumption metrics extracted from different circuit designs under varying operational conditions. The models analyzed include linear regression, ridge regression, lasso regression, random forest, gradient boosting, MLP regressor, XGBoost, LightGBM, CatBoost, and support vector regression (SVR). Each model was assessed based on its ability to predict power consumption accurately, using mean absolute error (MAE), mean squared error (MSE), and R² score as evaluation metrics.
   
 The Table II compares the actual and predicted dynamic power values. The actual dynamic power represents the real measured power consumption, while the predicted dynamic power is the estimated value provided by each model. Most models predict values close to the actual dynamic power, but some, such as the MLP Regressor and LightGBM, show significant deviations.

 <img width="502" height="312" alt="image" src="https://github.com/user-attachments/assets/9f942491-0a34-4470-b9ea-87a0ec9e3230" />

 Among all the models tested, ridge regression exhibited the most accurate and stable power predictions across different circuit architectures. Traditional linear regression suffered from high variance due to multi colinearity in the input features, while lasso regression overly penalized certain coefficients, leading to suboptimal predictions.

 <img width="502" height="302" alt="image" src="https://github.com/user-attachments/assets/b2d4e3b8-6c60-4d8b-bceb-f6b3370ce4c0" />

 The Table III focuses on total power, comparing actual and predicted values for each model. The actual total power reflects the real measured consumption, while the predicted total power represents the model’s estimate. Similar to the f irst table, Linear Regression, Ridge Regression, and XGBoost perform well, with their predictions closely matching the actual values.

 <img width="421" height="135" alt="image" src="https://github.com/user-attachments/assets/6e475c90-0fa4-47a8-8f67-97d605b8b8e0" />

  Fig. 4. Power Prediction

   The model is trained and tested and the graphical user interface has been made and which shows the final power predicted output which contains total and dynamic power as shown Fig 4. which has the power values in watts. Here the value is shown for particular input details. The graphical user interface is made user friendly to upload the dataset and train on different models along with model results which helps to choose best model for prediction. The GUI developed provided a seamless workflow for digital circuit designers, enabling direct upload of datasets in Excel (or CSV) format. Through intuitive controls, users can select their preferred machine learning models as mentioned above directly from the interface. The GUI manages the entire training and testing process internally, automatically performing steps such as data preprocessing, model training, validation, and prediction. Upon selection, users receive immediate visual feedback, including performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score, as well as plots comparing actual versus predicted power values. This modular design empowers users without programming expertise to experiment with various models, evaluate prediction accuracy, and choose the optimal algorithm for their specific VLSI power estimation needs. The integrated platform not only accelerates design iterations but also supports rapid, datadriven decision-making for power-aware circuit optimization.

   <img width="493" height="292" alt="image" src="https://github.com/user-attachments/assets/6f63d80b-f86c-4342-813e-c0183e3c2460" />

  Fig. 5. Power Prediction Plot

  The results suggest that ridge regression can be used for the dataset provided and the prediction graph with actual is shown in Fig.5. By providing accurate power predictions, it enables optimization of power-aware design techniques such as clock gating, voltage scaling, and dynamic power management, ultimately leading to reduced energy consumption in digital circuits. Future work will explore combining ridge regression with neural network-based models to further enhance accuracy while maintaining computational efficiency.

  <img width="516" height="172" alt="image" src="https://github.com/user-attachments/assets/befaa4fe-6b46-4f18-8851-4602bc8b0567" />

  [4] :  R. M B, S. P. K R, P. Das and A. Acharyya, ”GLAAPE: Graph Learning
 Assisted Average Power Estimation for Gate-level Combinational De
signs,” 2022 29th IEEE International Conference on Electronics, Circuits
 and Systems (ICECS), Glasgow, United Kingdom, 2022, pp. 1-4..

   The Table IV is a comparative study of prediction accuracy of various models for a 4-bit adder circuit. The MAE of the Random Forest model, as documented in earlier research, stands at 2.3, whereas that of the Ridge Regression model, developed in this paper, is much lower at 0.21. This decrease in error proves the higher precision and dependability of the proposed Ridge Regression method in power estimation. The outcome shows that Ridge Regression performs better than Random Forest in this particular task, and thus is a better option when it comes to accurate power estimation in digital circuit design. 

  CONCLUSION

   The paper introduced a machine learning-based solution to precise power estimation in VLSI circuits, with an emphasis on the comparison of multiple regression and ensemble learning models. Of the compared models, ridge regression outperformed the others through efficient management of multicollinearity and the presentation of realistic power estimates with little computational overhead. The Ridge Regression model significantly outperforms the Random Forest model in terms of mean absolute error (MAE) with respect to power estimation in digital circuits. For the estimation of powerin a 4-bit adder, Ridge Regression is at an MAE of 0.21 while Random Forest is at an MAE of 2.3, thus leading to an error reduction of around 90.87%. The results reflect the high level of precision and consistency offered by ridge regression in terms of estimating power. The new approach allows for the estimation at early stages, which is vital in optimizing the circuit design while carrying out low power design and early power prediction. The future work will construct hybrid schemes that blend the advantage of Ridge Regression with the capability of deep learning in handling high-order nonlinearities in VLSI circuit information. The new scheme allows for better flexibility and precision in diversified circuit designs. The generalizability is also boosted while overfitting is decreased. The scheme is adaptable even with new features or increased sizes in the datasets. Such hybrid schemes can be implemented effectively in the current GUI, allowing both fast inference and high accuracy in the tasks of power estimation. 
