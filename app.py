from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import os

app = Flask(__name__)
model = pickle.load(open("flight_rf.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

def calculate_price_for_date(model, features, base_date, days_offset):
    """Calculate price for a date with offset from base date, keeping all other features the same"""
    # Create a copy of features to avoid modifying original
    new_features = features.copy()

    # Create a new date by adding offset days to the base date
    new_dep_date = base_date + pd.Timedelta(days=days_offset)

    # Update date-related features
    new_features[1] = new_dep_date.day
    new_features[2] = new_dep_date.month

    # Make prediction with the model
    prediction = model.predict([new_features])[0]
    return {
        'price': round(prediction, 2),
        'date': new_dep_date.strftime('%Y-%m-%d'),
        'day_offset': days_offset
    }



@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            # Get input values from form
            date_dep = request.form["Dep_Time"]
            date_arr = request.form["Arrival_Time"]

            # Convert to datetime format
            dep_datetime = pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M")
            arr_datetime = pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M")

            # Validation: Departure must be before Arrival
            if dep_datetime >= arr_datetime:
                return render_template("home.html", prediction_text="Error: Arrival time must be later than Departure time.")

            # Extract features
            Journey_day = dep_datetime.day
            Journey_month = dep_datetime.month
            Dep_hour = dep_datetime.hour
            Dep_min = dep_datetime.minute
            Arrival_hour = arr_datetime.hour
            Arrival_min = arr_datetime.minute

            # Calculate duration
            duration = arr_datetime - dep_datetime
            dur_hour = duration.seconds // 3600
            dur_min = (duration.seconds // 60) % 60

            # Stopage
            Total_stops = int(request.form["stops"])

            # Airline Encoding
            airline = request.form['airline']
            airline_dict = {"IndiGo": 0, "Air India": 1, "SpiceJet": 2, "Vistara": 3, "GoAir": 4, "Vistara Premium economy": 5}
            airline_encoded = [0] * 11
            if airline in airline_dict:
                airline_encoded[airline_dict[airline]] = 1

            # Source Encoding
            Source = request.form["Source"]
            Destination = request.form["Destination"]
            source_dict = {"Delhi": 0, "Kolkata": 1, "Mumbai": 2, "Chennai": 3}
            destination_dict = {"Delhi": 0, "Cochin": 1, "Hyderabad": 2, "Kolkata": 3}
            source_encoded = [0] * 4
            destination_encoded = [0] * 5

            if Source in source_dict:
                source_encoded[source_dict[Source]] = 1
            if Destination in destination_dict:
                destination_encoded[destination_dict[Destination]] = 1

            # Feature Vector
            features = [
                Total_stops, Journey_day, Journey_month, Dep_hour, Dep_min,
                Arrival_hour, Arrival_min, dur_hour, dur_min,
                *airline_encoded, *source_encoded, *destination_encoded
            ]

            # Make Prediction
            prediction = model.predict([features])[0]
            output = round(prediction, 2)

                        # Price trend analysis (past 3 days, today, future 3 days)
            base_date = pd.Timestamp(year=dep_datetime.year, month=dep_datetime.month, day=dep_datetime.day)
            date_prices = []

            # Past 3 days
            for i in range(-3, 0):
                date_prices.append(calculate_price_for_date(model, features, base_date, i))

            # Today (current day)
            date_prices.append({
                'price': output,
                'date': base_date.strftime('%Y-%m-%d'),
                'day_offset': 0
            })

            # Future 3 days
            for i in range(1, 4):
                date_prices.append(calculate_price_for_date(model, features, base_date, i))

            lowest_price = min(date_prices, key=lambda x: x['price'])

            return render_template('home.html',
                                prediction_text=f"Your Flight price is Rs. {output}",
                                lowest_price=lowest_price['price'],
                                lowest_price_date=lowest_price['date'],
                                show_lowest_price=lowest_price['price'] < output)

        except Exception as e:
            return render_template("home.html", prediction_text=f"Error: {str(e)}")

    return render_template("home.html")


@app.route("/calendar", methods=["POST"])
@cross_origin()
def calendar():
    if request.method == "POST":
        try:
            date_dep = request.form["Dep_Time"]
            date_arr = request.form["Arrival_Time"]

            dep_datetime = pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M")
            arr_datetime = pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M")

            Journey_day = dep_datetime.day
            Journey_month = dep_datetime.month
            Dep_hour = dep_datetime.hour
            Dep_min = dep_datetime.minute
            Arrival_hour = arr_datetime.hour
            Arrival_min = arr_datetime.minute

            dur_hour = (arr_datetime - dep_datetime).seconds // 3600
            dur_min = ((arr_datetime - dep_datetime).seconds // 60) % 60

            Total_stops = int(request.form["stops"])
            airline = request.form['airline']

            airline_dict = {"IndiGo": 0, "Air India": 1, "SpiceJet": 2, "Vistara": 3,
                           "Air Asia": 4, "GoAir": 5, "Vistara Premium economy": 6}

            airline_encoded = [0] * 11
            if airline in airline_dict:
                airline_encoded[airline_dict[airline]] = 1

            Source = request.form["Source"]
            Destination = request.form["Destination"]

            source_dict = {"Delhi": 0, "Kolkata": 1, "Mumbai": 2, "Chennai": 3}
            destination_dict = {"Cochin": 0, "Delhi": 1, "Hyderabad": 2, "Kolkata": 3}

            source_encoded = [0] * 4
            if Source in source_dict:
                source_encoded[source_dict[Source]] = 1

            destination_encoded = [0] * 5
            if Destination in destination_dict:
                destination_encoded[destination_dict[Destination]] = 1

            features = [
                Total_stops, Journey_day, Journey_month, Dep_hour, Dep_min,
                Arrival_hour, Arrival_min, dur_hour, dur_min,
                *airline_encoded, *source_encoded, *destination_encoded
            ]

            # Calculate prices for calendar view (±3 days)
            base_date = pd.Timestamp(year=dep_datetime.year, month=dep_datetime.month, day=dep_datetime.day)
            date_prices = []

            for i in range(-3, 4):
                date_prices.append(calculate_price_for_date(model, features, base_date, i))

            lowest_price = min(date_prices, key=lambda x: x['price'])

            # Format date ranges for display
            date_range_start = (base_date - pd.Timedelta(days=3)).strftime('%d %b, %Y')
            date_range_end = (base_date + pd.Timedelta(days=3)).strftime('%d %b, %Y')

            # Get route info for display
            route_info = f"{Source} to {Destination}"

            return render_template('calendar.html',
                                  date_prices=date_prices,
                                  lowest_price=lowest_price,
                                  route_info=route_info,
                                  date_range_start=date_range_start,
                                  date_range_end=date_range_end)

        except Exception as e:
            return render_template("home.html", prediction_text=f"Error generating calendar: {str(e)}")


if __name__ == "__main__":
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))  # Get port from Render, default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=False)  # Bind to all interfaces




