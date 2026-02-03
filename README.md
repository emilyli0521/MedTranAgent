# MedTranAgent
A customer service agent for Medical Translation Company

MedTranAgent is a **workflow-based conversational agent** for **medical translation quotation**.
It is designed to prioritize deterministic business logic, controlled dialogue flow, and reliability over free-form conversation.

---

## Features

- **Stateful conversation**
  - Remembers user-provided information across turns
  - (document type, language pair, word count, urgency)

- **Strict intake gate**
  - Actively asks for missing required information
  - Prevents incomplete or hallucinated quotations

- **Deterministic pricing**
  - All pricing and turnaround time are generated via tools only
  - No manual calculation or model guessing

- **Tool chaining**
  - Pricing estimation → pricing formula explanation → service proposal summary

- **Hybrid interaction model**
  - Workflow-driven quotation process
  - Allows general questions (e.g. company policy) without breaking the workflow

- **Streaming support**
  - Token-level streaming responses for non-workflow questions

---

## Required Information Fields

The agent will continue asking until all required fields are provided:

- Document type (ICF / Protocol / IFU)
- Language direction (zh→en / en→zh)
- Word count
- Urgency / deadline

---

## How It Works

1. User starts describing a translation request
2. The agent collects missing required fields step by step
3. Once all required information is complete:
   - The agent asks whether to proceed with quotation
4. Pricing is calculated strictly via tools
5. Users can still ask general questions without leaving the quotation flow

---

## Project Structure
├── agent.py # Main conversational agent (workflow + state machine)
├── tools.py # Pricing and proposal tools
├── requirements.txt
├── README.md

## How to Run

### 1. Install dependencies
```bash
git clone https://github.com/your-username/MedTranAgent.git
cd MedTranAgent

### 2. Install dependencies
```bash
It is recommended to use a virtual environment.
pip install -r requirements.txt

### 3. Set environment variables
```env
Create a .env file in the project root directory and add your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

### 4. Run the agent
```bash
python agent.py
