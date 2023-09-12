import pandas as pd
import csv


def calculateData():
    insulinData = pd.read_csv("InsulinData.csv", parse_dates=[["Date", "Time"]])
    autoModeRows = insulinData[insulinData["Alarm"] == "AUTO MODE ACTIVE PLGM OFF"]
    autoModeRows = autoModeRows.sort_values(['Date_Time'], ascending=[True])
    autoModeActiveDateTime = autoModeRows.iloc[0]["Date_Time"]

    cgmData = pd.read_csv("CGMData.csv", parse_dates=[["Date", "Time"]])
    # print(cgmData.info())

    cgmData = cgmData[~cgmData["Sensor Glucose (mg/dL)"].isnull()]
    cgmData = cgmData.sort_values(['Date_Time'], ascending=[True])

    begDate = cgmData.iloc[0, 0].date()
    endDate = cgmData.iloc[len(cgmData) - 1, 0].date()
    dateRange = pd.date_range(start=begDate, end=endDate)

    datesWithRequiredReadings = []

    for date in dateRange:
        #     midnightBegDate = i
        #     midnightEndDate = i + pd.Timedelta("05:59:59")
        #     midnightReadings = autoCgmData[(autoCgmData["Date_Time"] >= midnightBegDate)
        #                                    & (autoCgmData["Date_Time"] <= midnightEndDate)
        #                                    & (~autoCgmData["Sensor Glucose (mg/dL)"].isnull())]

        #     daytimeBegDate = i + pd.Timedelta("06:00:00")
        #     daytimeEndDate = i + pd.Timedelta("23:59:59")
        #     dayReadings = autoCgmData[(autoCgmData["Date_Time"] >= daytimeBegDate)
        #                               & (autoCgmData["Date_Time"] <= daytimeEndDate)
        #                               & (~autoCgmData["Sensor Glucose (mg/dL)"].isnull())]

        todayBeg = date
        todayEnd = date + pd.Timedelta("23:59:59")
        currDateReadings = cgmData[(cgmData["Date_Time"] >= todayBeg) & (cgmData["Date_Time"] <= todayEnd)]

        if (len(currDateReadings) >= 231):
            datesWithRequiredReadings.append(date.date())

    cgmData["Date"] = cgmData["Date_Time"].dt.date
    requiredCgmData = cgmData[cgmData['Date'].isin(datesWithRequiredReadings)]

    autoCgmData = requiredCgmData[requiredCgmData["Date_Time"] >= autoModeActiveDateTime]
    autoModeMetrics = []
    autoModeMetrics.extend(getMetrics(autoCgmData, "overnight"))
    autoModeMetrics.extend(getMetrics(autoCgmData, "daytime"))
    autoModeMetrics.extend(getMetrics(autoCgmData, "fullday"))

    manualCgmData = requiredCgmData[cgmData["Date_Time"] < autoModeActiveDateTime]
    manualModeMetrics = []
    manualModeMetrics.extend(getMetrics(manualCgmData, "overnight"))
    manualModeMetrics.extend(getMetrics(manualCgmData, "daytime"))
    manualModeMetrics.extend(getMetrics(manualCgmData, "fullday"))

    metrics = []
    metrics.append(manualModeMetrics)
    metrics.append(autoModeMetrics)

    output = 'Results.csv'

    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in metrics:
            writer.writerow(row)


def getMetrics(myCgmData, duration):
    begDate = myCgmData.iloc[0, 0].date()
    endDate = myCgmData.iloc[len(myCgmData) - 1, 0].date()

    dateRange = pd.date_range(start=begDate, end=endDate)
    #     print(dateRange)

    hyperglycemiaDays = 0
    hyperglycemiaCriticalDays = 0
    rangePrimaryDays = 0
    rangeSecondaryDays = 0
    hypoglycemiaL1Days = 0
    hypoglycemiaL2Days = 0

    hyperglycemiaPercent = 0
    hyperglycemiaCriticalPercent = 0
    rangePrimaryPercent = 0
    rangeSecondaryPercent = 0
    hypoglycemiaL1Percent = 0
    hypoglycemiaL2Percent = 0

    for date in dateRange:
        #     midnightBegDate = i
        #     midnightEndDate = i + pd.Timedelta("05:59:59")
        #     midnightReadings = autoCgmData[(autoCgmData["Date_Time"] >= midnightBegDate)
        #                                    & (autoCgmData["Date_Time"] <= midnightEndDate)
        #                                    & (~autoCgmData["Sensor Glucose (mg/dL)"].isnull())]

        #     daytimeBegDate = i + pd.Timedelta("06:00:00")
        #     daytimeEndDate = i + pd.Timedelta("23:59:59")
        #     dayReadings = autoCgmData[(autoCgmData["Date_Time"] >= daytimeBegDate)
        #                               & (autoCgmData["Date_Time"] <= daytimeEndDate)
        #                               & (~autoCgmData["Sensor Glucose (mg/dL)"].isnull())]

        if duration == "overnight":
            begTimeStamp = date
            endTimeStamp = date + pd.Timedelta("05:59:59")
        if duration == "daytime":
            begTimeStamp = date + pd.Timedelta("06:00:00")
            endTimeStamp = date + pd.Timedelta("23:59:59")
        if duration == "fullday":
            begTimeStamp = date
            endTimeStamp = date + pd.Timedelta("23:59:59")

        currDateReadings = myCgmData[
            (myCgmData["Date_Time"] >= begTimeStamp) & (myCgmData["Date_Time"] <= endTimeStamp)]

        #     print("date: {}, midnightreads: {} -- daytimeReads: {} -- totalreads: {}"
        #           .format(i, len(midnightReadings), len(dayReadings), len(currDateReadings))

        hyperglycemia = 0
        hyperglycemiaCritical = 0
        rangePrimary = 0
        rangeSecondary = 0
        hypoglycemiaL1 = 0
        hypoglycemiaL2 = 0

        for i in range(len(currDateReadings)):
            currReading = currDateReadings.iloc[i]["Sensor Glucose (mg/dL)"]
            if 180 < currReading:
                hyperglycemia = hyperglycemia + 1

            if 250 < currReading:
                hyperglycemiaCritical = hyperglycemiaCritical + 1

            if 70 <= currReading <= 180:
                rangePrimary = rangePrimary + 1

            if 70 <= currReading <= 150:
                rangeSecondary = rangeSecondary + 1

            if currReading < 70:
                hypoglycemiaL1 = hypoglycemiaL1 + 1

            if currReading < 54:
                hypoglycemiaL2 = hypoglycemiaL2 + 1

        if hyperglycemia > 0:
            hyperglycemiaDays = hyperglycemiaDays + 1

        if hyperglycemiaCritical > 0:
            hyperglycemiaCriticalDays = hyperglycemiaCriticalDays + 1

        if rangePrimary > 0:
            rangePrimaryDays = rangePrimaryDays + 1

        if rangeSecondary > 0:
            rangeSecondaryDays = rangeSecondaryDays + 1

        if hypoglycemiaL1 > 0:
            hypoglycemiaL1Days = hypoglycemiaL1Days + 1

        if hypoglycemiaL2 > 0:
            hypoglycemiaL2Days = hypoglycemiaL2Days + 1

        hyperglycemiaPercent = hyperglycemiaPercent + (hyperglycemia / 288)
        hyperglycemiaCriticalPercent = hyperglycemiaCriticalPercent + (hyperglycemiaCritical / 288)
        rangePrimaryPercent = rangePrimaryPercent + (rangePrimary / 288)
        rangeSecondaryPercent = rangeSecondaryPercent + (rangeSecondary / 288)
        hypoglycemiaL1Percent = hypoglycemiaL1Percent + (hypoglycemiaL1 / 288)
        hypoglycemiaL2Percent = hypoglycemiaL2Percent + (hypoglycemiaL2 / 288)

    # print(hyperglycemiaDays)
    # print(hyperglycemiaCriticalDays)
    # print(rangePrimaryDays)
    # print(rangeSecondaryDays)
    # print(hypoglycemiaL1Days)
    # print(hypoglycemiaL2Days)

    if hyperglycemiaDays > 0:
        hyperglycemiaPercent = hyperglycemiaPercent / hyperglycemiaDays

    if hyperglycemiaCriticalDays > 0:
        hyperglycemiaCriticalPercent = hyperglycemiaCriticalPercent / hyperglycemiaCriticalDays

    if rangePrimaryDays > 0:
        rangePrimaryPercent = rangePrimaryPercent / rangePrimaryDays

    if rangeSecondaryDays > 0:
        rangeSecondaryPercent = rangeSecondaryPercent / rangeSecondaryDays

    if hypoglycemiaL1Days > 0:
        hypoglycemiaL1Percent = hypoglycemiaL1Percent / hypoglycemiaL1Days

    if hypoglycemiaL2Days > 0:
        hypoglycemiaL2Percent = hypoglycemiaL2Percent / hypoglycemiaL2Days

    # print("hyperglycemiaPercent: {}".format(hyperglycemiaPercent))
    # print("hyperglycemiaCriticalPercent: {}".format(hyperglycemiaCriticalPercent))
    # print("rangePrimaryPercent: {}".format(rangePrimaryPercent))
    # print("rangeSecondaryPercent: {}".format(rangeSecondaryPercent))
    # print("hypoglycemiaL1Percent: {}".format(hypoglycemiaL1Percent))
    # print("hypoglycemiaL2Percent: {}".format(hypoglycemiaL2Percent))

    metrics = [hyperglycemiaPercent, hyperglycemiaCriticalPercent, rangePrimaryPercent, rangeSecondaryPercent,
            hypoglycemiaL1Percent, hypoglycemiaL2Percent]
    metrics = [i * 100 for i in metrics]

    return metrics


if __name__ == '__main__':
    calculateData()
