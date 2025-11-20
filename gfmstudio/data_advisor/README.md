# geospatial-studio-data-advisor

You need sentinelhub credentials in a `.env` file:
```bash
SH_CLIENT_ID=.....
SH_CLIENT_SECRET=.....
```

This endpoint retrieves unique dates and corresponding data results for collections listed.
If a user defines more than one collection, the dates found for the first collection in the list are used as the primary dates and only matching dates found in the other collections are returned, with an allowance of days defined in `pre-days` and `post-days`. When no data is found for the dates defined, the nearest available dates are returned. If a user defines more than one collection, only nearest vailable dates matching in all collections are returned.

## Usage

To query data availability:
```json
{
  "collections": ["string"],
  "dates": ["string"],
  "bbox": [
    ["float"]
  ],
  "area_polygon": "string",
  "maxcc": 0,
  "pre_days": "int",
  "post_days": "int",
}
```

SAMPLE:
```json
{
  "collections": ["hls_l30", "s2_l2a"],
  "dates": ["2025-01-01_2025-01-07"],
  "bbox": [[-119.00578922913463, 34.037720731413295, -118.45572476113966, 34.299909448158225]],
  "maxcc": 90.0,
  "pre_days": 1,
  "post_days": 1
}
```

where:
* `data_connector` -  The data source to query. Currently supports 'sentinelhub', 'nasa_earthdata' and 'sentinel_aws'.
* `collections` -  The list of collections to query. For example s2_l2a, hls_l30, etc.
* `dates` -  The list of dates to search for data availability. The dates must be in the format YYYY-mm-dd with a range being <start_date>_<end_date>
* `bbox`  or `area_polygon` (one of) -  Bounding box coordinates in the format [min_lon, min_lat, max_lon, max_lat] or a polygon object.
* `maxcc`(Optional) -  Maximum cloud cover in `%`.
* `pre_days`(Optional) -  The number of days before the primary dates to consider.
* `post_days`(Optional) -  The number of days after the primary dates to consider.

SAMPLE RESPONSE (Where data was found):
```json
{
  "results": [
    {
      "bbox": [
        -119.00578922913463,
        34.037720731413295,
        -118.45572476113966,
        34.299909448158225
      ],
      "unique_dates": [
        "2025-01-02",
        "2025-01-07",
        "2025-01-06",
        "2025-01-05"
      ],
      "available_data": [
        {
          "id": "HLS.S30.T11SLT.2025007T183649.v2.0",
          "properties": {
            "datetime": "2025-01-07T18:45:03.298Z",
            "eo:cloud_cover": 34
          }
        },
        {
          "id": "HLS.S30.T11SLU.2025007T183649.v2.0",
          "properties": {
            "datetime": "2025-01-07T18:44:48.821Z",
            "eo:cloud_cover": 20
          }
        },
        {
          "id": "HLS.L30.T11SLT.2025006T182824.v2.0",
          "properties": {
            "datetime": "2025-01-06T18:28:24.479Z",
            "eo:cloud_cover": 0
          }
        },
        {
          "id": "HLS.L30.T11SLU.2025006T182824.v2.0",
          "properties": {
            "datetime": "2025-01-06T18:28:24.479Z",
            "eo:cloud_cover": 2
          }
        },
        {
          "id": "HLS.S30.T11SLT.2025005T184751.v2.0",
          "properties": {
            "datetime": "2025-01-05T18:54:55.801Z",
            "eo:cloud_cover": 1
          }
        },
        {
          "id": "HLS.S30.T11SLU.2025005T184751.v2.0",
          "properties": {
            "datetime": "2025-01-05T18:54:42.252Z",
            "eo:cloud_cover": 4
          }
        },
        {
          "id": "HLS.L30.T11SLT.2025005T183436.v2.0",
          "properties": {
            "datetime": "2025-01-05T18:34:36.875Z",
            "eo:cloud_cover": 0
          }
        },
        {
          "id": "HLS.L30.T11SLU.2025005T183412.v2.0",
          "properties": {
            "datetime": "2025-01-05T18:34:12.988Z",
            "eo:cloud_cover": 11
          }
        },
        {
          "id": "HLS.S30.T11SLT.2025002T183751.v2.0",
          "properties": {
            "datetime": "2025-01-02T18:45:02.738Z",
            "eo:cloud_cover": 35
          }
        },
        {
          "id": "HLS.S30.T11SLU.2025002T183751.v2.0",
          "properties": {
            "datetime": "2025-01-02T18:44:48.259Z",
            "eo:cloud_cover": 1
          }
        },
        {
          "id": "S2B_MSIL2A_20250107T183649_N0511_R027_T11SLT_20250107T215733",
          "properties": {
            "datetime": "2025-01-07T18:45:03Z",
            "eo:cloud_cover": 36.05
          }
        },
        {
          "id": "S2B_MSIL2A_20250107T183649_N0511_R027_T11SLU_20250107T215733",
          "properties": {
            "datetime": "2025-01-07T18:44:48Z",
            "eo:cloud_cover": 9.65
          }
        },
        {
          "id": "S2A_MSIL2A_20250105T184751_N0511_R070_T11SLT_20250105T221248",
          "properties": {
            "datetime": "2025-01-05T18:54:55Z",
            "eo:cloud_cover": 26.62
          }
        },
        {
          "id": "S2A_MSIL2A_20250105T184751_N0511_R070_T11SLU_20250105T221248",
          "properties": {
            "datetime": "2025-01-05T18:54:42Z",
            "eo:cloud_cover": 22.96
          }
        },
        {
          "id": "S2A_MSIL2A_20250102T183751_N0511_R027_T11SLT_20250102T221646",
          "properties": {
            "datetime": "2025-01-02T18:45:02Z",
            "eo:cloud_cover": 3.2
          }
        },
        {
          "id": "S2A_MSIL2A_20250102T183751_N0511_R027_T11SLU_20250102T221646",
          "properties": {
            "datetime": "2025-01-02T18:44:48Z",
            "eo:cloud_cover": 0.86
          }
        }
      ]
    }
  ]
}
```

SAMPLE RESPONSE (Where no data was found):
```json
{
  "results": [
    {
      "bbox": [
        -119.00578922913463,
        34.037720731413295,
        -118.45572476113966,
        34.299909448158225
      ],
      "unique_dates": [],
      "message": "The modalities s2_l2a do not have any data for the selected dates. Try the Bef_Days: 2024-12-31, 2025-01-02, 2025-01-05 or Aft_Days: 2025-01-07, 2025-01-10, 2025-01-12"
    }
  ]
}
```