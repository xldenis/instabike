import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

vehicle_data_file = "../../data/crashstatdata/TBL_VEHICLE_1995_2009_20111020.csv"
crash_data_file = "../../data/crashstatdata/TBL_CRASHES_1995_2009_20111020.csv"
borough_list = {'Brooklyn': 0, 'Manhattan': 1, 'Queens': 2, 'Staten Island': 3, 'The Bronx': 4}


def extract_xy(dataitems):
    X = []
    y = []
    for key, data in dataitems:
        for borough_idx in borough_list.values():
            new_x = [
                    data["month"],
                    data["day_of_week"],
                    borough_idx,
                    data["weather"]
                    ]
            new_y = data["borough_counts"][borough_idx]
            X.append(new_x)
            y.append(new_y)
    return X, y


def get_predictions_by_borough(test_data_items, borough, predictor):
    test_data_items = sorted(test_data_items, key=lambda d: d[0])
    dates = [d[0] for d in test_data_items]
    print len(dates)
    test_X, test_y = extract_xy(test_data_items)
    print len(test_X)
    # Filter for the relevant borough
    test_X = [x for x in test_X if x[2] == borough]
    print len(test_X)
    pred_y = []
    for x in test_X:
        pred_y.append(predictor.predict(x))
    return dates, pred_y

def moving_avg(l, i, window=10):
    subarray = l[i:i+window]
    return float(sum(subarray)) / len(subarray)

def moving_avg_list(l, window=10):
    ret_list = []
    for i in xrange(len(l)):
        ret_list.append(moving_avg(l, i))

    return ret_list

def avr(true_y, predicted_y):
    true_y = np.array(true_y)
    predicted_y = np.array(predicted_y)

    mean = true_y.mean()
    diff = true_y - predicted_y
    diff_sq = diff ** 2
    denom = true_y - mean
    denom_sq = denom ** 2

    return diff_sq.sum() / denom_sq.sum()


def main():

    # Get all the bike crashes
    with open(vehicle_data_file) as f:
        reader = csv.reader(f)
        bike_case_nums = set()
        for row in reader:
            body_type = row[5]
            if body_type == "35":
                case_num = row[1]
                bike_case_nums.add(case_num)

    raw_data = []
    with open(crash_data_file) as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            case_num = row[1]
            if case_num in bike_case_nums:
                raw_data.append(row)

    dataset = {}

    skipped = 0
    for row in raw_data:
        date_string = row[4]
        date = datetime.strptime(date_string, "%m/%d/%Y")
        dict_data = dataset.get(date, {})

        # Update count
        cur_count = dict_data.get("count", 0)
        dict_data["count"] = cur_count + 1

        # Update weather_reports
        weather_reports = dict_data.setdefault("weather", [])
        row_weather = row[18]
        weather_reports.append(row_weather)

        # Add borough counts
        borough_of_data = row[45]
        if borough_of_data not in borough_list:
            skipped += 1
            # Don't add this data point
            continue

        counts_by_boro = dict_data.setdefault("borough_counts", [0] * 5)
        borough_idx = borough_list[borough_of_data]
        counts_by_boro[borough_idx] += 1

        # Add month and day of week
        dict_data["day_of_week"] = date.weekday()
        dict_data["month"] = date.month

        dataset[date] = dict_data

    print "Total %d data points skipped" % skipped

    weather_states = [str(i) for i in range(7)]

    def get_majority_weather(weather_list):
        counts = {state: 0 for state in weather_states}
        for state in weather_states:
            counts[state] = weather_list.count(state)

        return max(counts.items(), key=lambda x: x[1])[0]

    for key in dataset:
        weather_list = dataset[key]["weather"]
        majority_weather = get_majority_weather(weather_list)
        dataset[key]["weather"] = majority_weather

    dataset_items = sorted(dataset.items(), key=lambda d:d[0])

    split = int(len(dataset_items) * .8)
    train_data, test_data = dataset_items[:split], dataset_items[split:]


    train_X, train_y = extract_xy(train_data)
    test_X, test_y = extract_xy(test_data)

    # from sklearn.ensemble import RandomForestRegressor
    # reg = RandomForestRegressor()
    from sklearn import svm
    reg = svm.SVR()

    reg.fit(train_X, train_y)
    pred_y = reg.predict(test_X)

    print "Arv score: ", avr(test_y, pred_y)

    dklist = dataset.items()
    dklist = sorted(dklist, key=lambda x:x[0])

    dates = [x[0] for x in dklist]
    boro0_count = [x[1]["borough_counts"][0] for x in dklist]

    plt.plot(dates, moving_avg_list(boro0_count), marker=".", linestyle="")

    test_dates, boro0_pred_counts = get_predictions_by_borough(test_data, 0, reg)
    plt.plot(test_dates, boro0_pred_counts, marker=".", linestyle="", color="r")
    plt.show()



if __name__ == '__main__':
    main()

