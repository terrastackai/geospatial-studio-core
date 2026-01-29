# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import traceback
from datetime import datetime, timedelta

import pandas as pd
from terrakit import DataConnector

from gfmstudio.inference.v2.schemas import DataAdvisorRequestSchema
from gfmstudio.log import logger


def get_nearest_data_available_dates(
    data_connector: DataConnector,
    collection,
    datetime_from,
    datetime_to,
    bbox=None,
    area_polygon=None,
    maxcc=80,
):
    """
    Function used to search for dates with data.
    The horizon for search is expanded up to 4X before and 4X after user selected date range
    If we don't find data for a span of about 2 months we return to user with no data response

    Args:
        data_details (dict): collection details
        collection (dict): data collection type
        datetime_from (str): start date of user submitted period
        datetime_to (str): end date of user submitted period
        bbox (list(float)): bbox
        area_polygon (str): area polygon
        maxcc (float): max cloud cover allowed
    """
    date_start = pd.to_datetime(datetime_from)
    date_end = pd.to_datetime(datetime_to)
    nearest_day_range = int(os.getenv("SH_NEAREST_DAY_RANGE", "7"))
    search_for_data_available_dates = True
    iterations_for_data_search = 0
    before_period_items_found = None
    after_period_items_found = None

    dc = data_connector

    # Try to find data available days
    # In those three tries we will expand the range by 2X; so we start with 7 then 14 then 28 days so going to nearly
    # one month back and one month ahead.
    # If after three trials we still can't find anything; Revert with no data found for location.

    while search_for_data_available_dates and iterations_for_data_search < 3:
        logger.debug(
            f"Searching available data days using {nearest_day_range} days extension"
        )
        # Boundaring dates before
        date_adjusted_from = date_start - timedelta(days=nearest_day_range)
        date_adjusted_from_str = date_adjusted_from.strftime("%Y-%m-%d")
        _, before_period_items_found = dc.connector.find_data(
            data_collection_name=collection,
            date_start=date_adjusted_from_str,
            date_end=datetime_from,
            area_polygon=area_polygon,
            bbox=bbox,
            maxcc=maxcc,
        )

        # Boundaring dates after
        current_date = pd.Timestamp.now()
        days_diff = (current_date - date_end).days
        if days_diff > 0 and days_diff < nearest_day_range:
            date_adjusted_to_str = current_date.strftime("%Y-%m-%d")
            _, after_period_items_found = dc.connector.find_data(
                data_collection_name=collection,
                date_start=datetime_to,
                date_end=date_adjusted_to_str,
                area_polygon=area_polygon,
                bbox=bbox,
                maxcc=maxcc,
            )
        elif days_diff > nearest_day_range:
            date_adjusted_to = date_end + timedelta(days=nearest_day_range)
            date_adjusted_to_str = date_adjusted_to.strftime("%Y-%m-%d")
            _, after_period_items_found = dc.connector.find_data(
                data_collection_name=collection,
                date_start=datetime_to,
                date_end=date_adjusted_to_str,
                area_polygon=area_polygon,
                bbox=bbox,
                maxcc=maxcc,
            )

        if not before_period_items_found and not after_period_items_found:
            iterations_for_data_search = iterations_for_data_search + 1
            nearest_day_range = nearest_day_range * 2
        else:
            search_for_data_available_dates = False

    # After exiting the while loop; check if we found any dates; otherwise alert user of no data
    if not before_period_items_found and not after_period_items_found:
        return {"before_options": [], "after_options": []}

    # If data available dates were found proceed to communicate with user
    before_options = []
    after_options = []

    # Get available dates before period selected by user
    if before_period_items_found is not None and len(before_period_items_found) > 0:
        before_period_datetimes = list(
            set(
                [
                    item_dict["properties"]["datetime"].split("T")[0]
                    for item_dict in before_period_items_found
                ]
            )
        )
        before_period_datetimes.sort()
        logger.debug(f"Nearest before days: {before_period_datetimes}")
        before_options = before_period_datetimes

    # Get available dates after period selected by user
    if after_period_items_found is not None and len(after_period_items_found) > 0:
        after_period_datetimes = list(
            set(
                [
                    item_dict["properties"]["datetime"].split("T")[0]
                    for item_dict in after_period_items_found
                ]
            )
        )
        after_period_datetimes.sort()
        logger.debug(f"Nearest after days: {after_period_datetimes}")
        after_options = after_period_datetimes

    return {
        "before_options": before_options,
        "after_options": after_options,
    }


def find_data_bbox(connector: DataConnector, payload: DataAdvisorRequestSchema):
    """
    This function retrieves unique dates and corresponding data results from a specified data source data collection.

    Args:
        connector str: The data source. For example sentinelhub, nasa_earthdata, sentinel_aws.
        payload (DataAdvisorRequestSchema): An object containing the necessary parameters for data retrieval.

    Returns:
        dict: A dictionary containing the unique dates and the results of the data retrieval.
    """

    input = payload
    collections = input["collections"]
    dates = input["dates"]
    bbox_list = input["bbox"]
    pre_days = input["pre_days"]
    post_days = input["post_days"]
    maxcc = input["maxcc"]

    dc = connector

    try:
        if bbox_list and all(len(x) == 4 for x in bbox_list):
            results = []
            for item in bbox_list:
                data_available = []
                alternative_dates_before = []
                alternative_dates_after = []
                collections_with_no_data = []
                collections_with_data = []
                for collection in collections:
                    res_dates = []
                    res_data = []
                    for date in dates:
                        if "_" in date:
                            date_start = date.split("_")[0]
                            date_end = date.split("_")[1]
                        else:
                            date_start = date
                            date_end = date
                        unique_dates, data = dc.connector.find_data(
                            data_collection_name=collection,
                            date_start=date_start,
                            date_end=date_end,
                            bbox=item,
                            maxcc=maxcc,
                        )
                        if unique_dates and data:
                            res_dates.extend(unique_dates)
                            res_data.extend(data)
                    if not res_data:
                        before_options = []
                        after_options = []
                        for date in dates:
                            if "_" in date:
                                date_start = date.split("_")[0]
                                date_end = date.split("_")[1]
                            else:
                                date_start = date
                                date_end = date
                            data_available_dates_options = (
                                get_nearest_data_available_dates(
                                    data_connector=dc,
                                    collection=collection,
                                    datetime_from=date_start,
                                    datetime_to=date_end,
                                    bbox=item,
                                    area_polygon=input["area_polygon"],
                                    maxcc=input["maxcc"],
                                )
                            )
                            returned_before_options = data_available_dates_options[
                                "before_options"
                            ]
                            if returned_before_options:
                                before_options.extend(returned_before_options)
                            returned_after_options = data_available_dates_options[
                                "after_options"
                            ]
                            if returned_after_options:
                                after_options.extend(returned_after_options)

                        alternative_dates_before.append(
                            list(dict.fromkeys(before_options))
                        )
                        alternative_dates_after.append(
                            list(dict.fromkeys(after_options))
                        )

                        collections_with_no_data.append(collection)
                    else:
                        collections_with_data.append(collection)
                    data_available.append(
                        {
                            "collector_name": collection,
                            "unique_dates": res_dates,
                            "available_data": res_data,
                        }
                    )

                if alternative_dates_before:
                    alternative_dates_before = list(
                        set(alternative_dates_before[0]).intersection(
                            *alternative_dates_before[1:]
                        )
                    )
                if alternative_dates_after:
                    alternative_dates_after = list(
                        set(alternative_dates_after[0]).intersection(
                            *alternative_dates_after[1:]
                        )
                    )

                # check data availability for alternative dates in other collections
                before_dates = []
                after_dates = []
                if collections_with_no_data:
                    if collections_with_data:
                        for modality in collections_with_data:
                            if alternative_dates_before:
                                for x in alternative_dates_before:
                                    a = datetime.strptime(x, "%Y-%m-%d") - timedelta(
                                        days=pre_days
                                    )
                                    b = datetime.strptime(x, "%Y-%m-%d") + timedelta(
                                        days=post_days
                                    )
                                    start_date = a.strftime("%Y-%m-%d")
                                    end_date = b.strftime("%Y-%m-%d")
                                    unique_dates_, data_ = dc.connector.find_data(
                                        data_collection_name=modality,
                                        date_start=start_date,
                                        date_end=end_date,
                                        bbox=item,
                                        maxcc=maxcc,
                                    )
                                    if data_:
                                        if x not in before_dates:
                                            before_dates.append(x)
                                    else:
                                        if x in before_dates:
                                            before_dates.remove(x)
                            if alternative_dates_after:
                                for y in alternative_dates_after:
                                    a = datetime.strptime(y, "%Y-%m-%d") - timedelta(
                                        days=pre_days
                                    )
                                    b = datetime.strptime(y, "%Y-%m-%d") + timedelta(
                                        days=post_days
                                    )
                                    start_date = a.strftime("%Y-%m-%d")
                                    end_date = b.strftime("%Y-%m-%d")
                                    unique_dates_2, data_2 = dc.connector.find_data(
                                        data_collection_name=modality,
                                        date_start=start_date,
                                        date_end=end_date,
                                        bbox=item,
                                        maxcc=maxcc,
                                    )
                                    if data_2:
                                        if y not in after_dates:
                                            after_dates.append(y)
                                    else:
                                        if y in after_dates:
                                            after_dates.remove(y)

                before_dates.sort()
                before_mydatetimes = ", ".join(before_dates)
                after_dates.sort()
                after_mydatetimes = ", ".join(after_dates)

                # date tolerance for dates found
                primary_dates = [
                    datetime.strptime(X, "%Y-%m-%d")
                    for X in data_available[0]["unique_dates"]
                ]
                if len(data_available) > 1:
                    combined_dates = []
                    other_dates = [
                        [datetime.strptime(X, "%Y-%m-%d") for X in Y["unique_dates"]]
                        for Y in data_available[1:]
                    ]
                    if len(primary_dates) > 0 and len(other_dates[0]) > 0:
                        for p in primary_dates:
                            od = other_dates[0]

                            pre_date = p - timedelta(days=pre_days)
                            post_date = p + timedelta(days=post_days)

                            time_diffs_tested = [
                                (
                                    (t - p)
                                    if (t >= pre_date) & (t <= post_date)
                                    else timedelta(days=100)
                                )
                                for t in od
                            ]
                            time_diffs_test = [
                                (t >= pre_date) & (t <= post_date) for t in od
                            ]
                            time_diffs_abs = [abs(X) for X in time_diffs_tested]

                            closest_index = time_diffs_abs.index(min(time_diffs_abs))

                            if time_diffs_test[closest_index] is True:
                                combined_dates = combined_dates + [
                                    p.strftime("%Y-%m-%d"),
                                    od[closest_index].strftime("%Y-%m-%d"),
                                ]
                        combined_dates = list(set(combined_dates))
                    elif len(other_dates[0]) == 0:
                        combined_dates = [D.strftime("%Y-%m-%d") for D in primary_dates]
                    else:
                        combined_dates = [
                            D.strftime("%Y-%m-%d") for D in other_dates[0]
                        ]
                    response = {
                        "bbox": item,
                        "unique_dates": combined_dates,
                        "available_data": [
                            val
                            for i in data_available
                            for val in i["available_data"]
                            if val
                            and datetime.fromisoformat(
                                val["properties"]["datetime"].replace("Z", "+00:00")
                            ).strftime("%Y-%m-%d")
                            in combined_dates
                        ],
                    }
                else:
                    response = {
                        "bbox": item,
                        "unique_dates": list(
                            set([x.strftime("%Y-%m-%d") for x in primary_dates])
                        ),
                        "available_data": data_available[0]["available_data"],
                    }
                if collections_with_no_data:
                    collections_with_no_data = ", ".join(collections_with_no_data)
                    response["unique_dates"] = []
                    response.pop("available_data")
                    response["message"] = (
                        f"The modalities {collections_with_no_data} do not have any data for the selected dates. "
                        f"Try the Bef_Days: {before_mydatetimes} or Aft_Days: {after_mydatetimes}"
                    )
                results.append(response)
            return {"results": results}
        elif input["area_polygon"] is not None and input["area_polygon"]:
            input["bbox"] = None
            unique_dates, results = dc.connector.find_data(
                data_collection_name=collection,
                date_start=date_start,
                date_end=date_end,
                area_polygon=input["area_polygon"],
                maxcc=maxcc,
            )
        else:
            raise Exception("bbox or area_polygon not provided")
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(
            f"Searching for available data raised error: {err}\nFull traceback:\n{tb_str}"
        )
        raise err
