import pickle
import numpy as np
import google.generativeai as gemini

gemini.configure(api_key="AIzaSyASUFBrNl_EsBuo8QD2_1HDGZXlcVAiG_o")

# Load the trained model from the pickle file
with open('HEART_MODEL/Heart Attack/heart_attack_prediction.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

# Function to make a prediction with new data
def make_prediction(input_data):
    prediction = ensemble_model.predict(np.array(input_data).reshape(1, -1))[0]
    probability = ensemble_model.predict_proba(np.array(input_data).reshape(1, -1))[0][1]
    return prediction, probability

# Function to generate a prevention report based on risk and disease
def generate_prevention_report(risk, disease, age):
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
        -Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it.

    **Details:**
    Risk: {risk:.2f}%
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    try:
        response = gemini.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_output_tokens=1000
        )
        
        report = response.result if hasattr(response, 'result') else None
        
        if not report:
            print("The response from the API did not contain a result.")
        
        return report
    except Exception as e:
        print(f"An error occurred: {e}")

# Example interactive input
print("\n--- Predict New Input ---")
input_data = []

# Collect input data for the specified features
try:
    age = float(input("Enter age: "))
    sex = float(input("Enter sex (0 for female, 1 for male): "))
    cp = float(input("Enter chest pain type (0-3): "))
    trtbps = float(input("Enter resting blood pressure (in mm Hg): "))
    chol = float(input("Enter cholesterol level (in mg/dl): "))
    fbs = float(input("Enter fasting blood sugar (> 120 mg/dl, 1 for true; 0 for false): "))
    restecg = float(input("Enter resting electrocardiographic results (0-2): "))
    thalachh = float(input("Enter maximum heart rate achieved: "))
    exng = float(input("Enter exercise induced angina (1 = yes; 0 = no): "))
    oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
    slp = float(input("Enter the slope of the peak exercise ST segment (0-2): "))
    caa = float(input("Enter number of major vessels (0-3) colored by fluoroscopy: "))
    thall = float(input("Enter thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect): "))

    # Add input values to the list in order
    input_data.extend([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])

    # Making prediction based on user input
    prediction, prediction_prob = make_prediction(input_data)

    # Log the exact prediction and probability
    print(f"Predicted class: {prediction}")
    print(f"Predicted heart attack probability: {prediction_prob:.4f}")

    # Calculate risk percentage
    risk_percentage = prediction_prob * 100

    # Adjust the threshold to capture high-risk cases
    threshold = 30.0  # Lower threshold to capture more high-risk cases

    # Determine if the person is at high risk of a heart attack
    if risk_percentage >= threshold:
        disease_type = "Heart Attack"
        print(f"Risk: High risk of heart attack")
        print(f"Risk Percentage: {risk_percentage:.2f}%")
        print(f"Problem: {disease_type}")

        # Generate the wellness report using the risk and disease information
        report = generate_prevention_report(
            risk=risk_percentage,
            disease=disease_type,
            age=age
        )

        if report:
            print("\nGenerated Wellness Report:")
            print(report)
        else:
            print("Failed to generate a report. Please check the API response and try again.")
    else:
        print(f"Risk: Low risk of heart attack")
        print(f"Risk Percentage: {risk_percentage:.2f}%")
        print("No report will be generated as the risk is low.")

except Exception as e:
    print(f"An error occurred during prediction: {e}")
