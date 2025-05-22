# Auto MPG Project: **Predicting Car Fuel Efficiency**

## Project Overview:

This project aims to predict the fuel efficiency (MPG) of cars using various features such as **displacement**, **horsepower**, **weight**, and others. The dataset used is the **Auto MPG dataset** from the **UCI Machine Learning Repository**.

The model used in this project is **Linear Regression**, which is trained to predict fuel efficiency based on the given features.

## Project Contents:

1. **Data Loading and Preprocessing**:

   * The **Auto MPG dataset** is loaded, and missing or erroneous values (e.g., in the `horsepower` column) are handled using appropriate methods.
   * The data is split into training and testing sets.

2. **Data Analysis**:

   * A **correlation matrix** is generated to explore the relationships between features and the target variable (MPG).
   * **Scatter plots** are used to visualize the relationships between features and the target.

3. **Modeling and Evaluation**:

   * A **Linear Regression** model is trained on the training data.
   * The model is evaluated using **Mean Squared Error (MSE)** and **R² Score** metrics.

4. **Results and Analysis**:

   * The model results are displayed, and the analysis focuses on which features influence fuel efficiency the most.

## How to Run the Project:

To run this project locally, first, install the necessary packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then, run the Python code with the following command:

```bash
python auto_mpg_project.py
```

## Dataset Details:

This project uses the **Auto MPG** dataset, which contains various features of cars, such as **cylinders**, **displacement**, **horsepower**, **weight**, **acceleration**, **model\_year**, **origin**, and **car\_name**.

### Features:

* **mpg**: Fuel efficiency (Miles Per Gallon)
* **cylinders**: Number of cylinders in the car's engine
* **displacement**: Engine displacement (in cubic inches)
* **horsepower**: Engine horsepower
* **weight**: Car weight
* **acceleration**: Car acceleration (in seconds for 0 to 60 mph)
* **model\_year**: Model year of the car
* **origin**: Country of origin (1: USA, 2: Europe, 3: Japan)
* **car\_name**: Name of the car

## Results:

The **Linear Regression** model performed reasonably well in predicting **mpg**. Feature analysis showed that variables such as **weight** and **horsepower** had the most significant negative impact on fuel efficiency.

## Possible Improvements:

To improve the model’s accuracy, alternative models like **Lasso Regression** or **Decision Trees** could be tried. Additionally, **Polynomial Regression** could be used to model nonlinear relationships between features and target variables.

## Setup and Run:

1. Download the project from GitHub.
2. Set up a Python environment and install the necessary packages.
3. Run the code and observe the results.

## Contributing:

If you have suggestions or changes to improve the project, feel free to submit a **Pull Request** or mention them in the **Issues** section.
