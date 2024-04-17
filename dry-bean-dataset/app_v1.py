import gradio as gr
import joblib
import pandas as pd

# Function to predict input using the model
def predict_input(
    area, perimeter, major_axis_length,
    minor_axis_length, aspect_ratio, eccentricity,
    convex_area, equiv_diameter, extent, solidity,
    roundness, compactness, shape_factor1,
    shape_factor2, shape_factor3, shape_factor4
):
    # Create a dictionary with the input data
    data = {
        'Area': area,
        'Perimeter': perimeter,
        'MajorAxisLength': major_axis_length,
        'MinorAxisLength': minor_axis_length,
        'AspectRation': aspect_ratio,  # Change the key to match the expected name
        'Eccentricity': eccentricity,
        'ConvexArea': convex_area,
        'EquivDiameter': equiv_diameter,
        'Extent': extent,
        'Solidity': solidity,
        'roundness': roundness,  # Change the key to match the expected name
        'Compactness': compactness,
        'ShapeFactor1': shape_factor1,
        'ShapeFactor2': shape_factor2,
        'ShapeFactor3': shape_factor3,
        'ShapeFactor4': shape_factor4
    }
    
    # Create a DataFrame with the input data
    X_inp = pd.DataFrame([data])
    
    # Load the saved model and encoder
    clf = joblib.load('dry-bean-dataset/saved_model.pkl')
    enc = joblib.load('dry-bean-dataset/saved_encoder.pkl')
    
    # Predict the class
    y_pred = clf.predict(X_inp)
    
    # Return the bean class name (inverse transform prediction)
    return enc.inverse_transform(y_pred)[0]

# Create a Gradio interface
ui = gr.Interface(
    fn=predict_input,
    inputs=[
        gr.Number(label='Area'),
        gr.Number(label='Perimeter', step=1),
        gr.Number(label='Major Axis Length', step=1),
        gr.Number(label='Minor Axis Length', step=1),
        gr.Number(label='Aspect Ratio', step=1),
        gr.Number(label='Eccentricity', step=1),
        gr.Number(label='Convex Area'),
        gr.Number(label='Equivalent Diameter', step=1),
        gr.Number(label='Extent', step=1),
        gr.Number(label='Solidity', step=1),
        gr.Number(label='Roundness', step=1),
        gr.Number(label='Compactness', step=1),
        gr.Number(label='Shape Factor 1', step=1),
        gr.Number(label='Shape Factor 2', step=1),
        gr.Number(label='Shape Factor 3', step=1),
        gr.Number(label='Shape Factor 4', step=1)
    ],
    outputs='text',
    title="Dry Bean Classification",
    examples=None  # Replace `df.iloc[:5].values.tolist()` with your examples if available
)

# Launch the Gradio interface
ui.launch()
