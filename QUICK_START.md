# Quick Start Guide - Text-to-SQL Fine-tuning Framework

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Setup
```bash
python test_setup.py
```

### 3. Setup Database & Generate Data
```bash
python database_setup.py
python data_generator.py
```

### 4. Start Training
```bash
python finetune_model.py
```

### 5. Evaluate & Demo
```bash
python evaluate_model.py --comprehensive
python demo.py
```

### 6. Test Visualization
```bash
python test_visualization.py
python visualization.py
```

## 📁 Project Structure

```
text_to_sql/
├── config.py              # Configuration parameters
├── utils.py               # Utility functions
├── database_setup.py      # Database creation
├── data_generator.py      # Training data generation
├── finetune_model.py      # Model training
├── evaluate_model.py      # Model evaluation
├── demo.py               # Interactive demo
├── visualization.py      # Data visualization
├── test_setup.py         # Setup verification
├── test_visualization.py # Visualization testing
├── requirements.txt      # Dependencies
└── README.md            # Full documentation
```

## 🎯 What This Framework Does

This framework fine-tunes **Llama 3.1 8B** to convert natural language financial queries to SQL with >80% accuracy.

### Example Queries:
- "Show me Apple's closing prices for the last 30 days"
- "Calculate Tesla's volatility over the past month"
- "Find the best performing tech stock this quarter"
- "What is the correlation between Tesla and Apple movements?"

## ⚡ Performance Targets

- **Syntax Accuracy**: >90% valid SQL
- **Execution Rate**: >85% queries run without errors
- **Semantic Accuracy**: >80% correct results
- **Training Time**: 3-4 hours on M3 MacBook Air

## 🔧 Hardware Requirements

- **Recommended**: MacBook Air M3 (optimized for Apple Silicon)
- **Minimum**: 8GB RAM, 10GB free disk space
- **Alternative**: Any system with Python 3.8+

## 📊 Database Schema

```sql
CREATE TABLE stock_data (
    Date TEXT,           -- YYYY-MM-DD format
    Open REAL,           -- Opening price
    High REAL,           -- High price
    Low REAL,            -- Low price
    Close REAL,          -- Closing price
    Adj_Close REAL,      -- Adjusted closing price
    Volume INTEGER,      -- Trading volume
    ticker TEXT,         -- Stock symbol
    move REAL            -- Daily % change
);
```

## 🎮 Interactive Usage

### Command Line Demo
```bash
python demo.py --interactive
```

### Visualization Demo
```bash
python visualization.py
python demo.py --demo  # Includes plotting
```

### Data Visualization Features
- **Price History**: Plot stock prices over time
- **Returns Distribution**: Analyze daily returns
- **Volatility Comparison**: Compare rolling volatility
- **Correlation Heatmaps**: Visualize stock correlations
- **Volume Analysis**: Price and volume charts
- **SQL Results**: Auto-plot query results

### Programmatic Usage
```python
from evaluate_model import SQLEvaluator
from visualization import FinancialDataVisualizer

# Generate SQL
evaluator = SQLEvaluator("financial_sql_model")
sql = evaluator.generate_sql("Show me Tesla's volatility last month")

# Visualize results
viz = FinancialDataVisualizer()
viz.plot_sql_query_results(sql, plot_type='line')
```

## 🔍 Evaluation Modes

### Comprehensive Evaluation
```bash
python evaluate_model.py --comprehensive
```

### Interactive Testing
```bash
python evaluate_model.py --interactive
```

### Custom Model Path
```bash
python evaluate_model.py --model_path /path/to/model
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **Memory Issues**: Reduce batch size in `config.py`
3. **Slow Training**: Ensure you're using Apple Silicon (M1/M2/M3)
4. **Database Errors**: Run `python database_setup.py`

### Performance Tips

- Use synthetic data for faster testing: `use_real_data=False`
- Reduce training steps: Change `max_steps` in `config.py`
- Use smaller batch size: Reduce `per_device_train_batch_size`

## 📈 Expected Results

After training, you should see:
- SQL queries that execute successfully
- Natural language understanding of financial terms
- Proper handling of date ranges and aggregations
- Multi-stock comparison capabilities

## 🎓 Advanced Usage

### Custom Training Data
```python
from data_generator import FinancialSQLGenerator

generator = FinancialSQLGenerator()
custom_data = generator.generate_training_data(5000)  # 5000 samples
```

### Custom Evaluation
```python
from evaluate_model import SQLEvaluator

evaluator = SQLEvaluator("my_model")
custom_queries = ["Your custom query here"]
results = evaluator.evaluate_test_set(custom_queries)
```

## 📞 Support

If you encounter issues:
1. Run `python test_setup.py` to diagnose problems
2. Check the error messages for specific issues
3. Ensure all dependencies are installed correctly
4. Verify sufficient disk space and memory

## 🎉 Success Metrics

You'll know it's working when:
- ✅ Training completes without errors
- ✅ Evaluation shows >80% execution rate
- ✅ Interactive demo responds to financial queries
- ✅ Generated SQL executes successfully

---

**Happy fine-tuning! 🚀** 