# Installation Guide

## Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- (Optional) Ollama for LLM features
- (Optional) Interactive Brokers account for live trading

## Step 1: Clone Repository

```bash
git clone https://github.com/shkomig/Trading_System.git
cd Trading_System
```

## Step 2: Create Virtual Environment

### Windows
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Linux/Mac
```bash
python -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

### Basic Installation (for backtesting and analysis)
```bash
pip install pandas numpy matplotlib seaborn plotly pyyaml python-dotenv loguru tqdm psutil joblib
```

### Full Installation (includes ML models)
```bash
pip install -r requirements.txt
```

**Note**: TensorFlow/PyTorch installation may take time and requires sufficient disk space.

### Minimal Installation (if you want to skip ML models)
```bash
pip install pandas numpy matplotlib seaborn plotly pyyaml python-dotenv loguru tqdm psutil joblib scikit-learn ta pandas-ta sqlalchemy pytest streamlit
```

## Step 4: Configuration

### 4.1 Create `.env` File

```bash
# Copy example env file
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
IB_ACCOUNT=DU1234567

# Email Notifications (optional)
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USERNAME=your@email.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENTS=recipient@email.com

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789

# Ollama LLM (optional)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### 4.2 Review Configuration Files

Check and modify if needed:
- `config/config.yaml` - Main system configuration
- `config/strategies.yaml` - Strategy parameters (if exists)

## Step 5: Verify Installation

### Run System Info
```bash
python src/main.py --mode info
```

Expected output:
```
========================================================================
                   AUTOMATED TRADING SYSTEM                          
========================================================================

Available Strategies (8):
  ✓ ma_crossover
  ✓ triple_ma
  ...
```

### Run Tests (Optional)
```bash
pytest tests/ -v
```

## Step 6: Optional Components

### 6.1 Install Ollama (for LLM features)

**Windows**:
1. Download Ollama from https://ollama.ai
2. Install and run
3. Pull a model: `ollama pull llama2`

**Linux/Mac**:
```bash
curl https://ollama.ai/install.sh | sh
ollama pull llama2
```

Verify:
```bash
ollama list
```

### 6.2 Setup Interactive Brokers (for live trading)

1. Download TWS or IB Gateway from Interactive Brokers
2. Enable API connections:
   - TWS: Configure → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Set port (default: 7497 for paper, 7496 for live)
3. Start TWS/IB Gateway
4. Update `.env` with your account details

**Important**: Start with **Paper Trading** account!

## Step 7: Run Examples

### Simple Backtest
```bash
python examples/simple_backtest.py
```

### ML Models Demo (requires TensorFlow)
```bash
python examples/ml_models_example.py
```

### Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

The dashboard will open in your browser at http://localhost:8501

## Troubleshooting

### Issue: TensorFlow installation fails

**Solution**: Try installing TensorFlow separately:
```bash
# For CPU-only version
pip install tensorflow-cpu

# Or skip ML models for now
```

### Issue: "Module not found" errors

**Solution**: Ensure you're in the project root and virtual environment is activated:
```bash
cd Trading_System
# Activate venv
python -c "import sys; print(sys.executable)"  # Should show venv path
```

### Issue: Ollama connection error

**Solution**: 
1. Verify Ollama is running: `ollama list`
2. Check port in `.env` matches Ollama server
3. LLM features will be disabled if Ollama is unavailable

### Issue: "Permission denied" on Windows

**Solution**: Run PowerShell as Administrator or adjust execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Database errors

**Solution**: Delete and recreate database:
```bash
rm data/trading_system.db
python src/data/database.py  # Recreates database
```

## Next Steps

After successful installation:

1. **Read Documentation**: Check `README.md` and `QUICK_START.md`
2. **Run Backtests**: Test strategies on historical data
3. **Paper Trading**: Connect to IB paper account
4. **Live Trading**: Only after thorough testing!

## System Requirements

### Minimum
- CPU: 2 cores
- RAM: 4 GB
- Disk: 2 GB free space

### Recommended
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 10+ GB free space (for ML models and data)
- SSD for better performance

## Support

If you encounter issues:

1. Check logs in `logs/` directory
2. Run with verbose logging: `python src/main.py --verbose`
3. Review documentation: `README.md`, `QUICK_START.md`
4. Check GitHub Issues: https://github.com/shkomig/Trading_System/issues

---

**Happy Trading! 📈💰**

