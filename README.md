# Policy Optimization for Financial Decision-Making

**Machine Learning Engineer Take-Home Assignment**  
**Submitted for**: Shodh AI  
**Author**: Shreyansh Singh  
**Contact**: shreyansh31456@gmail.com  
**GitHub**: [@Shreyansh-HelloWorld](https://github.com/Shreyansh-HelloWorld)

---

## Project Overview

This project compares two approaches to automated loan approval decisions:
1. **Deep Learning Model**: Neural network that predicts default probability (PyTorch)
2. **Reinforcement Learning Agent**: Policy that directly optimizes for profit

**Main Result**: The RL agent earns **$54 more per loan** than the DL model by being more selective. It approves 17% fewer loans, but those loans have much better quality (83.7% repayment vs 78.5% baseline). The small per-loan advantage adds up to $5.62M in additional profit on the 105,000-loan test set.

The key insight: **predictive accuracy and business profitability are different objectives** that lead to different decisions. The models disagree on 17,845 loans (17.0%), and analyzing these disagreements shows why optimizing for the right metric matters.

---

## Dataset

**Source**: [LendingClub Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

**Details**:
- Original: 2.26 million loan applications (2007-2018)
- Completed loans: 1.37 million (after filtering pending/current status)
- Sample: 700,000 loans (51% of completed loans - substantial portion)
- Features: 21 (after removing data leakage features)
- Target: Binary {0: Fully Paid, 1: Defaulted}
- Split: 70% train (490K) / 15% validation (105K) / 15% test (105K)
- Class balance: 78.5% Paid, 21.5% Default (maintained via stratified sampling)

---

## Project Structure

```
loan-policy-optimization/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── notebooks/                        # All analysis notebooks
│   ├── 01_EDA_Preprocessing.ipynb    # Task 1: Data exploration
│   ├── 02_Deep_Learning_Model.ipynb  # Task 2: Neural network
│   ├── 03_RL_Agent.ipynb             # Task 3: RL policy
│   └── 04_Analysis_Comparison.ipynb  # Task 4: Model comparison
│
├── models/                           # Model summaries and predictions
│   ├── dl_model_summary.json         # DL performance metrics
│   ├── rl_agent_summary.json         # RL performance metrics
│   ├── dl_predictions.csv            # DL predictions on test set
│   └── rl_predictions.csv            # RL predictions on test set
│
├── figures/                          # Visualizations (13 total)
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── dl_roc_curve.png
│   ├── rl_feature_importance.png
│   └── decision_agreement_analysis.png
│
└── reports/
    └── Final_Report.pdf             
```

**Note**: Raw data (`.csv.gz`), trained model weights (`.pt`, `.pkl`), and processed data files are not included due to file size. The `requirements.txt` enables full reproduction.

---

## Setup Instructions

### Requirements
- Python 3.9+
- 8GB RAM (16GB recommended)
- 2GB free disk space
- MacOS (M1/M2/M3) or Linux

### Installation Steps

1. **Clone and navigate**
```bash
git clone https://github.com/Shreyansh-HelloWorld/loan-approval-optimization.git
cd loan-policy-optimization
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
# Option 1: Kaggle API (recommended)
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/raw/

# Option 2: Manual download
# Visit: https://www.kaggle.com/datasets/wordsforthewise/lending-club
# Place accepted_2007_to_2018Q4.csv.gz in data/raw/
```

5. **Launch Jupyter**
```bash
jupyter notebook
```

---

## Running the Analysis

Run the notebooks in order:

### Task 1: `01_EDA_Preprocessing.ipynb`
- Exploratory data analysis
- Feature selection (removed 18 data leakage features)
- Data cleaning and preprocessing
- Train/val/test split (70/15/15)
- **Runtime**: ~8 minutes
- **Output**: Processed data files, 5 visualizations

### Task 2: `02_Deep_Learning_Model.ipynb`
- PyTorch neural network (21→128→64→32→1)
- Training with early stopping
- Evaluation: AUC-ROC, F1-Score, confusion matrix
- **Runtime**: ~15-20 minutes on M3 (with MPS acceleration)
- **Output**: Trained model, predictions, 3 visualizations

### Task 3: `03_RL_Agent.ipynb`
- RL problem formulation (states, actions, rewards)
- Gradient boosting policy training (200 estimators)
- Expected policy value evaluation
- **Runtime**: ~3-4 minutes
- **Output**: Trained agent, predictions, 3 visualizations

### Task 4: `04_Analysis_Comparison.ipynb`
- Comparative analysis of DL vs RL
- Deep dive into 17,845 disagreement cases
- Grade-wise and financial impact analysis
- **Runtime**: ~2 minutes
- **Output**: Comparative visualizations, insights

---

## Results Summary

### Deep Learning Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| AUC-ROC | **0.7175** | Strong - matches best Kaggle benchmarks without data leakage |
| F1-Score | 0.1080 | Low due to class imbalance and threshold |
| Accuracy | 78.93% | Reasonable |
| Precision | 59.72% | When predicts default, it's right ~60% of the time |
| Recall | 5.94% | Very low - only catches 6% of actual defaults |
| Approval Rate | 97.9% | Extremely liberal |

**Key characteristic**: The model is very conservative about predicting "default" (only predicts it 1,339 times out of 105,000), which means it approves almost everything. This happens because of class imbalance (78.5% paid vs 21.5% default) and using threshold 0.5.

### Reinforcement Learning Agent Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Expected Value | **$5,022 per loan** | Main metric - average profit per loan |
| Total Profit | $527.31M | On 105,000 test loans |
| Approval Rate | 80.9% | More selective than DL |
| Loan Quality | 83.7% repayment | Better than 78.5% baseline |
| Training Time | 3.4 minutes | Fast! |

**Key characteristic**: The agent is more selective, denying 17% more applications than the DL model, but the loans it approves have significantly better repayment rates.

### Head-to-Head Comparison

| Metric | DL Model | RL Agent | Winner |
|--------|----------|----------|--------|
| Approval Rate | 97.9% | 80.9% | DL (approves more) |
| Profit per Loan | $4,968 | $5,022 | **RL (+$54)** |
| Total Profit | $521.68M | $527.31M | **RL (+$5.62M)** |
| Loan Quality | 78.4% paid | 83.7% paid | **RL (+5.3pp)** |
| Decision Agreement | - | 83.0% | 17.0% disagree |

### The 17,845 Disagreement Cases

These are loans the DL model approves but the RL agent denies - this is where the interesting findings are.

**What makes them different?**
- Larger loan amounts: $15,825 vs $15,663 (avg for agreed approvals)
- Higher interest rates: 15.8% vs 15.7%
- Worse credit metrics: Higher DTI, lower grades
- Higher default rate: 41.2% vs 16.3%

**Why does DL approve them?**
The DL model rarely predicts "default" (recall is only 5.9%). For these loans, it outputs default probabilities around 0.3-0.5, which is below the 0.5 threshold, so it approves them.

**Why does RL deny them?**
Even though 58.8% would pay, the expected value is negative:
```
Expected Value = 0.588 × $2,500 - 0.412 × $15,825 = -$1,725
```
The potential loss from the 41% who default is larger than the profit from the 59% who pay. The RL agent, directly optimizing for expected value, correctly identifies these as unprofitable.

**Business Impact**: By avoiding these 17,845 unprofitable loans, the RL agent earns the extra $5.62M.

---

## Technologies

- **Python 3.9+**: Core language
- **PyTorch 2.5**: Deep learning (with M3 MPS support)
- **Scikit-learn 1.5**: Preprocessing, metrics, gradient boosting
- **Pandas 2.2**: Data manipulation
- **NumPy 2.0**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive notebooks

---

## Report

The **[Final Report](reports/Final_Report.pdf)** (3 pages) provides detailed analysis including:
1. Introduction and problem framing
2. Dataset and preprocessing methodology
3. Deep learning model architecture and results
4. Reinforcement learning formulation and results
5. **Comparative analysis** - deep dive into disagreement cases
6. Limitations and future work
7. Conclusions and business implications

---

## Key Takeaways

### 1. Different Optimization Goals Lead to Different Decisions
- **DL**: Optimizes classification accuracy → Approves liberally (97.9%)
- **RL**: Optimizes expected profit → More selective (80.9%)

### 2. The $54 Per Loan Advantage Adds Up
Small improvements per loan scale significantly across thousands of decisions. In a real lending portfolio with millions of loans, this could be tens of millions in additional profit.

### 3. Accuracy ≠ Profitability  
The DL model achieves strong AUC-ROC (0.7175), matching benchmarks. But predictive accuracy doesn't automatically translate to business value. The RL agent demonstrates that directly optimizing for the business objective yields better outcomes.

### 4. Grade-Wise Patterns
Agreement is high for low-risk loans (Grade A-C). Disagreement increases dramatically for high-risk loans - for Grade F and G, the RL agent is much more selective about loans that DL approves.

### 5. Broader Application
This approach isn't limited to lending. Any domain with business objectives can benefit:
- Healthcare: Optimize patient outcomes, not just diagnosis accuracy
- Marketing: Optimize customer lifetime value, not just click-through rate
- Supply chain: Optimize total cost, not just forecast error

---

## Limitations

1. **Simplified reward function** - assumes full repayment or full default (no partial payments)
2. **Static historical data** - trained on 2007-2018; may not reflect current conditions
3. **No online learning** - models don't adapt over time
4. **Single RL algorithm** - only tested gradient boosting; others might perform better
5. **Threshold sensitivity** - DL performance could improve with optimal threshold (0.1997 vs 0.5)

---

## Future Work

**Near-term improvements:**
- Test DL model at optimal threshold for fair comparison
- Ensemble multiple RL agents
- Incorporate loan term duration into reward
- Add temporal validation (train on earlier years, test on later years)

**Research directions:**
- Causal inference to estimate true treatment effects
- Fairness-aware RL with demographic parity constraints
- Contextual bandits for continuous learning with exploration
- Multi-objective optimization (profit + risk + fairness)
- Explainability via SHAP values or counterfactuals

---

## Development Notes

A few things I learned while building this:

**Technical:**
- M3 MPS acceleration made PyTorch training ~3x faster than CPU
- The class imbalance issue was trickier than expected - even with proper sampling, the model became very conservative about predicting defaults
- Feature importance from the RL agent revealed some non-obvious patterns (sub_grade matters most, but DTI and revolving balance are also critical)

**Insights:**
- Initially expected the RL agent to approve *more* high-risk loans (betting on high interest rates), but the opposite happened - it became more selective overall
- The disagreement analysis took longer than anticipated but revealed the most interesting findings
- The $54/loan difference seems tiny, but it's the *why* behind it that matters - demonstrates the importance of optimizing for the right metric

**If I were deploying this:**
I'd probably use a hybrid approach: DL model generates default probabilities → RL-style expected value calculation for final decision. This combines the interpretability of probabilities with business-focused optimization.

---

## Author

**Shreyansh Singh**  
Machine Learning Engineer

- Email: shreyansh31456@gmail.com
- GitHub: [@Shreyansh-HelloWorld](https://github.com/Shreyansh-HelloWorld)

---

## Acknowledgments

- LendingClub for providing the dataset
- Kaggle for hosting and maintaining the data
- Shodh AI for the interesting problem and opportunity

---

## References

1. LendingClub Dataset: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. PyTorch Documentation: https://pytorch.org/docs/
3. Scikit-learn Documentation: https://scikit-learn.org/
4. Sutton & Barto (2018): *Reinforcement Learning: An Introduction*
5. Levine et al. (2020): "Offline Reinforcement Learning: Tutorial, Review, and Perspectives"

---

**Last Updated**: October 2025
