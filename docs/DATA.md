# Dataset Documentation

## Files
We have train/test splits for 5 stocks:

- `data/AAPL_train.csv`, `data/AAPL_test.csv`
- `data/GS_train.csv`, `data/GS_test.csv`
- `data/JPM_train.csv`, `data/JPM_test.csv`
- `data/TSLA_train.csv`, `data/TSLA_test.csv`
- `data/XOM_train.csv`, `data/XOM_test.csv`

## Columns
Each file contains the following columns:

- **Date**: Trading day (YYYY-MM-DD)
- **Open, High, Low, Close, Volume**: Original OHLCV stock price data
- **LogReturn**: Daily log return  
  \[
  \text{LogReturn}_t = \ln \left(\frac{Close_t}{Close_{t-1}}\right)
  \]
- **MA5, MA10, MA20**: 5-day, 10-day, and 20-day moving averages of closing prices
- **RSI14**: 14-day Relative Strength Index

## Train/Test Split
- **Train** = all rows where `Date < 2019-01-01`  
- **Test** = all rows where `Date >= 2019-01-01`

This ensures models are trained on older data and evaluated on newer, unseen data.

---

## Notes
- Missing values were forward-filled (`ffill`) and back-filled (`bfill`) if necessary.
- Columns like `LogReturn`, MAs, and RSI may have NaN values in the first few rows due to rolling-window calculations.
- Data covers approx. 2010–2024 depending on stock availability.
