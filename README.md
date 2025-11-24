# Fraud Detection API

A production-ready fraud detection API that combines rule-based heuristics with local LLM (Llama 3.2) for intelligent, real-time transaction analysis.

## Overview

This API analyzes financial transactions and provides fraud predictions using an ensemble approach that combines:
- **Rule-Based Detection**: Fast, deterministic fraud detection using configurable thresholds
- **LLM Analysis**: Context-aware fraud detection using local Llama 3.2
- **Ensemble Methods**: Multiple strategies to combine both approaches for optimal accuracy

## Features

- **5 Detection Strategies**: Choose between rules-only, LLM-only, weighted ensemble, cascade, or maximum
- **Real-Time Analysis**: Sub-second response times for rule-based detection
- **Explainable AI**: Detailed reasoning for each fraud prediction
- **Local LLM**: Privacy-focused using Ollama with Llama 3.2
- **RESTful API**: Clean, well-documented FastAPI endpoints
- **Configurable**: Easy-to-adjust thresholds and weights
- **Production-Ready**: Comprehensive error handling and logging

## Architecture

```
┌─────────────────┐
│   Transaction   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ FastAPI │
    └────┬────┘
         │
    ┌────▼──────────────────┐
    │ Ensemble Detector     │
    ├───────────┬───────────┤
    │           │           │
┌───▼────┐  ┌──▼─────┐    │
│ Rules  │  │  LLM   │    │
│ Engine │  │(Llama) │    │
└───┬────┘  └──┬─────┘    │
    │          │           │
    └──────┬───┴───────────┘
           │
    ┌──────▼──────┐
    │   Response  │
    └─────────────┘
```

## Tech Stack

- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation
- **Ollama**: Local LLM runtime
- **Llama 3.2**: 3.2B parameter model
- **Python 3.10+**: Core language
- **httpx**: Async HTTP client

## Installation

### Prerequisites

1. **Python 3.10 or higher**
2. **Ollama** with Llama 3.2 model

#### Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.2
ollama pull llama3.2
```

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd fraud-detection-api
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
# or if using uv:
uv pip install -e .
```

4. **Verify Ollama is running**
```bash
ollama list  # Should show llama3.2
```

## Running the API

### Start the server

```bash
uvicorn main:app --reload --port 8001
```

The API will be available at:
- **API**: http://localhost:8001
- **Interactive Docs**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## API Usage

### Endpoint

```
POST /api/v1/predict?strategy={strategy}
```

### Detection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `rules_only` | Fast rule-based detection | High-volume, known patterns |
| `llm_only` | AI-powered analysis | Complex cases, need explanations |
| `weighted` ⭐ | Balanced hybrid (default) | Production default |
| `cascade` | Smart: rules first, LLM if uncertain | Cost optimization |
| `max` | Conservative: highest risk score | High-security scenarios |

### Request Example

```bash
curl -X POST "http://localhost:8001/api/v1/predict?strategy=weighted" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-2024-001234",
    "user_id": "USER-12345",
    "amount": 1200.00,
    "payment_method": "credit_card",
    "ip_address": "192.168.1.1",
    "country": "NG",
    "merchant_id": "MERCHANT-789",
    "device_id": null,
    "timestamp": "2024-01-15T02:30:00Z"
  }'
```

### Response Example

```json
{
  "transaction_id": "TXN-2024-001234",
  "is_fraud": true,
  "fraud_score": 0.77,
  "risk_level": "critical",
  "flags": [
    "high_amount",
    "high_risk_country",
    "unusual_hours",
    "missing_device_info"
  ],
  "ai_reasoning": "High-risk combination: large amount + suspicious location + unusual hour + unknown device. Pattern matches account takeover fraud.",
  "detection_method": "weighted",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Configuration

Edit `app/core/config.py` to adjust:

```python
# LLM Settings
ollama_base_url = "http://localhost:11434"
ollama_model = "llama3.2"
llm_temperature = 0.1  # Low for consistency

# Ensemble Weights
llm_weight = 0.6      # LLM contribution
rules_weight = 0.4    # Rules contribution

# Thresholds
fraud_threshold = 0.5       # Above = fraud
high_risk_threshold = 0.75  # Above = critical
```

Or use environment variables:
```bash
export OLLAMA_MODEL=llama3.2
export LLM_WEIGHT=0.6
export FRAUD_THRESHOLD=0.5
```

## Project Structure

```
fraud-detection-api/
├── app/
│   ├── api/
│   │   └── routes.py              # API endpoints
│   ├── models/
│   │   └── transaction.py         # Pydantic models
│   ├── services/
│   │   ├── fraud_detector.py      # Rule-based engine
│   │   ├── llm_fraud_detector.py  # LLM detector
│   │   ├── ensemble_detector.py   # Ensemble logic
│   │   └── prompt_builder.py      # Prompt engineering
│   └── core/
│       ├── config.py               # Configuration
│       └── llm_client.py           # Ollama client
├── main.py                         # FastAPI application
├── pyproject.toml                  # Dependencies
└── README.md
```

## Example Test Cases

### Legitimate Transaction
```json
{
  "transaction_id": "TXN-SAFE-001",
  "user_id": "USER-999",
  "amount": 45.99,
  "payment_method": "debit_card",
  "country": "US",
  "device_id": "DEVICE-KNOWN-123"
}
```
**Expected**: `fraud_score < 0.25`, `risk_level: "low"`

### Fraudulent Transaction
```json
{
  "transaction_id": "TXN-FRAUD-001",
  "user_id": "USER-666",
  "amount": 9999.00,
  "payment_method": "credit_card",
  "country": "NG",
  "device_id": null,
  "timestamp": "2024-01-15T03:00:00Z"
}
```
**Expected**: `fraud_score > 0.75`, `risk_level: "critical"`

## Performance

| Strategy | Avg Response Time | Accuracy | Cost |
|----------|------------------|----------|------|
| rules_only | ~10ms | Good | Low |
| llm_only | ~2-5s | Excellent | Medium |
| weighted | ~2-5s | Best | Medium |
| cascade | ~10ms - 5s | Very Good | Optimized |

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Port Already in Use
```bash
# Use a different port
uvicorn main:app --port 8002
```

### LLM Response Parsing Errors
- The system includes fallback parsing
- Check logs for details
- Adjust `llm_temperature` for more consistent outputs

## Development

### Run with auto-reload
```bash
uvicorn main:app --reload --port 8001
```

### Code Quality
```bash
# Format code
ruff format .

# Lint code
ruff check .
```

## Future Enhancements

- [ ] Database integration for transaction history
- [ ] User behavior profiling
- [ ] API rate limiting
- [ ] Webhook notifications
- [ ] Dashboard UI
- [ ] Model fine-tuning on custom data
- [ ] A/B testing framework
- [ ] Batch processing endpoint

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.
