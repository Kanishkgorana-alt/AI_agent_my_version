🚀 Customer Retention Learning Agent using AI






An AI-driven system that predicts customer churn and generates personalized retention offers.
The system learns from customer feedback (accept/reject) and continuously improves its decisions using a learning agent architecture.

📌 Project Description

Customer churn is a major problem in banking. This project goes beyond prediction by building an adaptive decision system that:

Predicts churn probability using a neural network
Identifies key features affecting churn
Generates targeted retention offers
Learns from feedback to improve future decisions

The system follows a continuous loop:

Observe → Decide → Act → Learn → Improve
⚙️ Installation Steps
1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Create virtual environment
python -m venv .venv
3. Activate environment

Windows

.venv\Scripts\activate

Linux / Mac

source .venv/bin/activate
4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
▶️ How to Run the Project

Start the Streamlit app:

streamlit run app.py

Open in browser:

http://localhost:8501
📥 Example Input / Output
Example Input
Customer Details:
- Credit Score: 650
- Geography: France
- Age: 40
- Balance: 60000
- Number of Products: 1
- Is Active Member: Yes
Example Output
Churn Probability: 0.78 (High Risk)

Selected Features to Improve:
- Balance
- Activity

Recommended Offer:
- Cashback bonus on card usage

Explanation:
Customer shows low engagement and moderate balance.
Cashback incentives may increase activity and retention.
📁 Project Structure
├── app.py
├── models/
├── utils/
├── logs/
│   ├── update_history.csv
│   └── parameters.csv
├── Churn_Modelling.csv
├── requirements.txt
🧠 Key Features
Machine learning-based churn prediction
Learning agent architecture
Feedback-driven updates
Personalized offer generation
Real-time Streamlit interface
🧾 Notes
Ensure dataset (Churn_Modelling.csv) is present
logs/ folder is required for storing updates
Run inside virtual environment for best results
📌 Summary

This project demonstrates how AI can move from prediction → decision-making → continuous learning, making it highly applicable for real-world customer retention systems.