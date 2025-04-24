# Startup Success Predictor

A machine learning application that predicts the probability of a startup's success based on funding, rounds, and business category.

## 📊 Features

- **Predictive Model**: Uses machine learning to predict startup success probability
- **Data Visualization**: Visualizes prediction results and model comparisons
- **Command-Line Interface**: Easy-to-use CLI for making predictions
- **Docker Support**: Containerized application for easy deployment

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker (optional, for containerized usage)

### Installation

#### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jaecy03/Startup-Success.git
   cd Startup-Success
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t startup-success-predictor .
   ```

## 💻 Usage

### Command Line Interface

Make a prediction for a startup:

```bash
python predict_cli.py --name "TechStartup" --funding 1000000 --rounds 2 --category "Software"
```

Generate a visualization of the prediction:

```bash
python predict_cli.py --name "TechStartup" --funding 1000000 --rounds 2 --category "Software" --visualize
```

Compare different machine learning models:

```bash
python predict_cli.py --compare-models
```

Save visualizations to a specific path:

```bash
python predict_cli.py --name "TechStartup" --funding 1000000 --rounds 2 --category "Software" --save-viz "path/to/save.png"
```

### Using Docker

Run the container with a prediction:

```bash
docker run startup-success-predictor predict_cli.py --name "TechStartup" --funding 1000000 --rounds 2 --category "Software"
```

Generate visualizations and save them to your local machine:

```bash
docker run -v $(pwd)/output:/app/visualizations startup-success-predictor predict_cli.py --name "TechStartup" --funding 1000000 --rounds 2 --category "Software" --visualize
```

## 📈 Visualization Examples

The application generates two types of visualizations:

1. **Model Comparison**: Compares the accuracy and standard deviation of different machine learning models.
2. **Prediction Visualization**: Shows the success probability of a startup with a gauge chart and pie chart.

Visualizations are saved to the `visualizations/` directory by default.

## 🧠 Machine Learning Models

The application evaluates several machine learning models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost

The best-performing model is automatically selected and used for predictions.

## 🐳 Docker Implementation

The project includes a Dockerfile that:

1. Uses Python 3.12 slim as the base image
2. Installs necessary system dependencies for matplotlib
3. Sets up a working directory at `/app`
4. Installs Python dependencies
5. Creates a directory for visualizations
6. Uses an ENTRYPOINT to allow flexible commands

Benefits of using Docker:
- Portability: Run the application on any system with Docker installed
- Reproducibility: Consistent environment and results
- Isolation: Avoid conflicts with other applications
- Dependency Management: Clean installation of all dependencies
- Deployment Ready: Ready for deployment to various environments

## 📁 Project Structure

```
startup-success/
├── data/                  # Dataset directory
│   └── startup_data.csv   # Startup dataset
├── visualizations/        # Generated visualizations
├── main.py                # Main application code
├── predict_cli.py         # Command-line interface
├── visualize.py           # Visualization module
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🛠️ Technologies Used

- **Python**: Core programming language
- **scikit-learn**: Machine learning models and preprocessing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **Docker**: Containerization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Jahnavi Sharma - [GitHub Profile](https://github.com/Jaecy03)

Project Link: [https://github.com/Jaecy03/Startup-Success](https://github.com/Jaecy03/Startup-Success)
