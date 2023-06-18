import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, flash
import io
import base64

app = Flask(__name__)
app.secret_key = '123'

@app.route("/regression")
def index():
    return render_template("index.html")

@app.route("/regression", methods=["POST", "GET"])
def regression():
    numeric_values_x_numinput = request.form['a']
    numeric_values_y_numinput = request.form['b']
    polyreg_value_numinput = request.form['c']
    graph_x_numinput = request.form['d']
    graph_y_numinput = request.form['e']

    # Getting numerical input from the user x_axis
    values = numeric_values_x_numinput.split(' ')
    numeric_values_x = [float(value) for value in values]

    # Getting numerical input from the user y_axis
    values = numeric_values_y_numinput.split(' ')
    numeric_values_y = [float(value) for value in values]

    # Getting numerical input from the user degree of polyregrsion
    polyreg_value = int(float(polyreg_value_numinput))

    # Getting numerical input from the user for starting point of graph on x-axis
    graph_x = int(float(graph_x_numinput))

    # Getting numerical input from the user for starting point of graph on y-axis
    graph_y = int(float(graph_y_numinput))

    # Applying polynomial regression on received data
    mymodel = np.poly1d(np.polyfit(numeric_values_x, numeric_values_y, polyreg_value))
    # Get the coefficients of the polynomial
    coefficients = mymodel.coeffs

    # Generate the equation string
    equation_str = "y = "
    for i, coeff in enumerate(coefficients[::-1]):
        power = polyreg_value - i
        equation_str += f"{coeff}x^{power} + "

    # Remove the trailing '+' sign and extra spaces
    equation_str = equation_str[:-3]

    myline = np.linspace(graph_x, graph_y)

    # Plotting the data and regression line
    plt.scatter(numeric_values_x, numeric_values_y)
    plt.plot(myline, mymodel(myline))
    
    # Saving the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Converting the plot image to base64 for embedding in HTML
    plot_image = base64.b64encode(buffer.getvalue())
    plot_image_uri = f"data:image/png;base64,{plot_image.decode()}"

    # Calculating the accuracy score
    accuracy_score = r2_score(numeric_values_y, mymodel(numeric_values_x))
    
    return render_template("index.html", plot_image_uri=plot_image_uri, accuracy_score=accuracy_score, equation_str=equation_str)


if __name__ == '__main__':
    app.run()

    
