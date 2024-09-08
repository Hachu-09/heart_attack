from flask import Flask, request, jsonify
import pickle
import numpy as np
import google.generativeai as gemini
import google.generativeai as genai


# Configure the Gemma API
gemini.configure(api_key="AIzaSyASUFBrNl_EsBuo8QD2_1HDGZXlcVAiG_o")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the Flask app
app = Flask(__name__)


# Load the trained heart attack model from the pickle file
with open('HEART_MODEL/Heart Attack/heart_attack_prediction.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

# Function to make a prediction with new data
def make_prediction(input_data):
    # Convert input data to numpy array and reshape for model prediction
    input_array = np.array(input_data).reshape(1, -1)
    
    # Predict heart attack risk and probability
    prediction = ensemble_model.predict(input_array)[0]
    probability = ensemble_model.predict_proba(input_array)[0][1]
    
    return prediction, probability

# Function to generate a prevention report based on risk, disease, and age
def generate_prevention_report(risk, disease, age):
    # Define the prompt for generating the wellness report
    prompt = f"""
   Provide a general wellness report with the following sections:

    1. **Introduction**
        -Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        -Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        -Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        -Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        -Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        -Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        -Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        -Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        -Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        -Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        -Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        -In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        -Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        -Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        -Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        -Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        -Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        -Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        -Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        -Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        -Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        -Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        -Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        -Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    10. **References**
        -Scientific Sources: Provide references to the scientific literature or authoritative health guidelines that support the information and recommendations given in the report.
        -Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it
    Risk: {risk:.2f}%
    Disease: {disease}
    Age: {age}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
    return None

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.get_json(force=True)
    
    # Check if input data exists
    if not data or 'input_data' not in data:
        return jsonify({'error': 'No input data provided'}), 400
    
    input_data = data['input_data']
    
    # Ensure input data has the expected format and size (13 features for heart attack prediction)
    if isinstance(input_data, dict):
        input_data = [
            input_data.get('age'),
            input_data.get('sex'),
            input_data.get('cp'),
            input_data.get('trtbps'),
            input_data.get('chol'),
            input_data.get('fbs'),
            input_data.get('restecg'),
            input_data.get('thalachh'),
            input_data.get('exng'),
            input_data.get('oldpeak'),
            input_data.get('slp'),
            input_data.get('caa'),
            input_data.get('thall')
        ]
    
    # Ensure the input list contains exactly 13 elements
    if not isinstance(input_data, list) or len(input_data) != 13:
        return jsonify({"error": "input_data must be a list of 13 numerical values"}), 400
    
    try:
        # Make the prediction based on input data
        prediction, probability = make_prediction(input_data)
        
        # Calculate risk percentage
        risk_percentage = probability * 100
        
        # Define the risk threshold for heart attack
        threshold = 30.0  # Consider risk significant if over 30%
        
        # Determine the disease type and generate the prevention report
        disease = "Heart Attack" if risk_percentage >= threshold else "No significant risk"
        age = input_data[0]
        prevention_report = generate_prevention_report(risk_percentage, disease, age)
        
        # Construct the response with prediction, probability, and prevention report
        result = {
            "prediction": int(prediction),
            "probability": risk_percentage,
            "disease": disease,
            "prevention_report": prevention_report
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
