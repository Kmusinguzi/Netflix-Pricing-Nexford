**Netflix Revenue \& Churn Model**



📋 **What This Does**

Predicts customer churn using Random Forest and optimizes pricing. Deploys to AWS SageMaker for real-time revenue predictions.



🚀 **3-Phase Setup**



**Phase 1: Local Prediction**

```python

\# Run churn analysis and train model

python netflix\_churn\_model.py

Input: Customer data CSV

Output: Trained model + revenue visualizations



**Phase 2: AWS Deployment**

Prerequisites:



AWS CLI configured (aws configure)



SageMaker permissions in IAM



S3 bucket access



python

\# Deploy to SageMaker

python sagemaker\_deployment.py

Output: Live endpoint URL



**Phase 3: API Invocation**

python

\# Test the deployed endpoint

python invoke\_endpoint.py

Sample Payload:



json

{

&nbsp; "subscribers": 2000000,

&nbsp; "avg\_watch\_hours": 5.5,

&nbsp; "monthly\_fee": 12.99,

&nbsp; "marketing\_spend": 200000

}

📁 **File Structure**

netflix\_churn\_model.py - Train Random Forest model



sagemaker\_deployment.py - Deploy to AWS



invoke\_endpoint.py - Call the live API



inference.py - SageMaker serving script



⚡ **Quick Test**

Run netflix\_churn\_model.py → Get local predictions



Run sagemaker\_deployment.py → Deploy to cloud



Run invoke\_endpoint.py → Test live API



Replace AWS credentials in deployment script before use.





