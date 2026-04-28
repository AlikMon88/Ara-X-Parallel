# << Parallel × Ara >>

### Agentic Semantic Debugging for ML Pipelines

## Overview

**Parallel × Ara** is a continuous ML debugging system that monitors training in real time and automatically performs **Root Cause Analysis (RCA)** when failures occur.

**Ara** provides persistent monitoring.
**Parallel** performs structured debugging using coordinated agents.

Together, they create an **autonomous debugging layer** for complex ML systems.

---

## Core Idea

Traditional ML debugging is manual.

Parallel × Ara turns debugging into an automated workflow:

```text
Training Runs
        ↓
Ara Monitors Logs
        ↓
Failure Detected
        ↓
Parallel Agents Triggered
        ↓
Root Cause Identified
        ↓
Fix Suggested
        ↓
Incident Stored
```

---

## System Architecture

Parallel uses **LangGraph** to orchestrate modular diagnostic agents.

Ara provides:

• Continuous monitoring
• Runtime scheduling
• Failure triggering

Parallel provides:

• Multi-tool diagnostics
• Structured reasoning
• Root cause generation

---

## Diagnostic Tools

Parallel includes specialized tools for:

• Log analysis
• Data distribution inspection
• Feature importance analysis
• Model architecture validation
• Historical incident retrieval (FAISS)

---

## Example Trigger

Ara detects training instability:

```json
{
  "epoch": 17,
  "is_trigger": true,
  "trigger_reason": "Loss became NaN"
}
```

Parallel automatically runs RCA and outputs through the streamlit frontend:


---

## Key Capabilities

• Continuous training monitoring
• Automated root cause analysis
• Multi-agent debugging workflows
• Experience-based failure memory
• Modular tool architecture

---

## Running Parallel

Standalone debugging:

```bash
python runner.py \
  --train_file_path=... \
  --stream_file_path=...
```

---

## Running Parallel × Ara

Start training:

```bash
python models/sample_train_2.py
```

Start monitoring:

```bash
python agent/ara_agents/monitor.py ## starts the API-rest
ngrok http <PORT> ## for API-serve-tunneling
```

Parallel will automatically trigger when failures are detected.

---

## Future Work

• Inference-time monitoring
• Concept drift detection
• Automated retraining
• RCA visualization dashboards

---

## License

MIT License.

---
